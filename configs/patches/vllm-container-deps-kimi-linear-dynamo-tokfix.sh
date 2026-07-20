#!/usr/bin/env bash
# Kimi-Linear dynamo-frontend tokenizer fix (for DISAGG / any frontend.type: dynamo).
# =============================================================================
# WHY: dynamo's frontend tokenizer (dynamo-tokenizers crate, tiktoken.rs) only
#   recognizes model_type in {kimi, kimi_k2, kimi_k25, deepseek_v3} and REJECTS
#   'kimi_linear' -> "Unsupported tiktoken model_type" -> /v1/chat/completions 404
#   -> aiperf warmup_failure. (AGG avoids this via routerless vllm-direct frontend;
#   DISAGG needs the dynamo router, so it must be fixed.)
#
# FIX: Kimi-Linear's tiktoken.model vocab is BYTE-IDENTICAL to kimi_k2
#   (verified: same HF LFS oid b6c497a7..., 2,795,286 bytes across Kimi-Linear /
#   Kimi-K2 / Kimi-K2-Thinking), so dynamo's KIMI_PATTERN BPE is exactly correct.
#   We spoof model_type -> kimi_k2 ONLY in dynamo's frontend tokenizer cache
#   (/root/.cache/dynamo/mdc/by-slug/...). The vLLM backend reads its own HF hub
#   cache where model_type stays kimi_linear, so the model architecture is
#   unaffected — the two caches are separate.
#   kimi_k2 (not kimi_k25) because Kimi-Linear-Instruct shares K2-Instruct's
#   tool-call format (tokenizer_config identical size); k25 is the reasoning variant.
#
# MECHANISM: this runs in the frontend node's bash preamble (setup_script). dynamo
#   retries model registration ~every 30s and RE-READS config.json from disk each
#   time, so a background poller that patches the mdc config.json makes the next
#   retry succeed. On backend/agg nodes the mdc dir never appears -> harmless no-op.
#
# STATUS: PROTOTYPE — validated logic (tokenizer identity + retry re-read), but not
#   yet run on a real disagg pipeline. Racy by design (relies on the retry window).
# =============================================================================
set -euo pipefail

echo "=== Kimi-Linear dynamo-tokfix: base deps + mdc model_type spoof (kimi_linear->kimi_k2) ==="

if [[ -f /configs/patches/vllm-container-deps.sh ]]; then
    bash /configs/patches/vllm-container-deps.sh
fi

# FIX: give dynamo a valid Kimi tokenizer.json so ModelDeploymentCard::from_disk picks
# TokenizerKind::HfTokenizerJson (which returns BEFORE the tiktoken path) -> the kimi_linear
# model_type check is skipped entirely. The bundled tokenizer.json is generated from Kimi's
# tiktoken.model + KIMI_PATTERN and byte-verified 11/11 vs the reference (identical token ids
# incl. specials). Root fix would be a 1-line dynamo whitelist add (tiktoken.rs:192).
TOKJSON=/tmp/kimi-linear-tokenizer.json
if [[ -f /configs/patches/kimi-linear-tokenizer.json.gz ]]; then
    gunzip -c /configs/patches/kimi-linear-tokenizer.json.gz > "$TOKJSON" 2>/dev/null \
      && echo "[dynamo-tokfix] staged tokenizer.json ($(wc -c <"$TOKJSON") B)"
fi

# SYNCHRONOUS pre-stage (wins the MDC race): the earlier background poller LOST — the backend
# content-hashes + registers the MDC the instant the engine is ready (tiktoken card 6f101c94),
# and the HF snapshot only materializes ~ms before that, so a 2s poller injected too late and
# into a different mdcsum dir. dynamo's tokenizer resolution is entirely container-local
# (/root/.cache/huggingface + /root/.cache/dynamo), NOT the shared HF_HOME=/lustre weights cache.
# So here, BEFORE the backend process starts, we synchronously pre-download the model's small
# (non-weight) files into the container-local HF cache and drop tokenizer.json into that snapshot.
# Then when the backend builds the MDC it reads tokenizer.json first -> HfTokenizerJson card ->
# frontend materializes HfTokenizerJson -> no tiktoken model_type check. No shared-cache pollution.
REPO="moonshotai/Kimi-Linear-48B-A3B-Instruct"
if [[ -s "$TOKJSON" ]]; then
    python3 - "$REPO" "$TOKJSON" <<'PYEOF' || echo "[dynamo-tokfix] pre-stage failed (poller fallback remains)"
import os, sys, shutil
repo, tokjson = sys.argv[1], sys.argv[2]
try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print(f"[dynamo-tokfix] huggingface_hub unavailable: {e}", flush=True); sys.exit(0)
# container-local dynamo cache is where the MDC tokenizer is resolved (per frontend logs)
caches = ["/root/.cache/huggingface/hub"]
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
for cache in caches:
    try:
        # NOTE: deliberately NOT fetching config.json — dynamo blake3-checksums it against the
        # registered MDC, so any config.json we write (even the identical HF copy, if it differs
        # from the backend's registered revision) risks a "checksum mismatch" (observed #58792797).
        # dynamo downloads config.json itself; we only add tokenizer.json into the same snapshot.
        p = snapshot_download(
            repo, cache_dir=cache, token=tok,
            allow_patterns=["tokenizer_config.json", "tiktoken.model",
                            "generation_config.json", "special_tokens_map.json", "*.py"],
        )
        dst = os.path.join(p, "tokenizer.json")
        if not os.path.exists(dst):
            shutil.copy(tokjson, dst)
        print(f"[dynamo-tokfix] PRE-STAGED tokenizer.json -> {dst} (before backend MDC build)", flush=True)
    except Exception as e:
        print(f"[dynamo-tokfix] pre-stage skip {cache}: {e}", flush=True)
PYEOF
fi

# Fallback poller (belt-and-suspenders for late/other cache dirs; the sync pre-stage above is primary).
# IMPORTANT: tokenizer.json ONLY — do NOT touch config.json. dynamo blake3-checksums config.json
# against the registered MDC, so any config.json mutation (the old model_type sed spoof) triggers
# "checksum mismatch for …/config.json" and re-fails registration (observed #58792797). tokenizer.json
# alone removes the tiktoken model_type check, and adding a file dynamo didn't register doesn't break
# the config.json checksum.
(
    for _ in $(seq 1 1800); do       # ~60 min, covers late-appearing cache/snapshot dirs
        while IFS= read -r tk; do
            d="$(dirname "$tk")"
            if [ -f "$TOKJSON" ] && [ ! -f "$d/tokenizer.json" ]; then
                cp "$TOKJSON" "$d/tokenizer.json" 2>/dev/null \
                  && echo "[dynamo-tokfix] injected tokenizer.json -> $d (HfTokenizerJson wins over tiktoken)"
            fi
        done < <(find /root/.cache/huggingface /root/.cache/dynamo -name tiktoken.model 2>/dev/null)
        sleep 2
    done
) </dev/null 2>&1 &
disown 2>/dev/null || true

echo "=== dynamo-tokfix poller backgrounded (inject tokenizer.json only; no config.json mutation) ==="
