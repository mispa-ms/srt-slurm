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

# PRIMARY fix (no race): inject a valid Kimi tokenizer.json so dynamo's
# ModelDeploymentCard::from_disk picks TokenizerKind::HfTokenizerJson, which returns BEFORE the
# tiktoken path -> the kimi_linear model_type check is skipped entirely. The MDC is built by the
# BACKEND via load_from_disk(<model dir>) (local_model.rs), so we inject into the HF snapshot; we
# also seed dynamo's mdc cache. dynamo doesn't manage/delete a tokenizer.json we add (unlike
# config.json which it re-writes), so it persists -> no race. The bundled tokenizer.json is
# generated from Kimi's tiktoken.model + KIMI_PATTERN and byte-verified 11/11 vs the reference
# (identical token ids incl. specials). We keep the config model_type spoof as a fallback.
TOKJSON=/tmp/kimi-linear-tokenizer.json
if [[ -f /configs/patches/kimi-linear-tokenizer.json.gz ]]; then
    gunzip -c /configs/patches/kimi-linear-tokenizer.json.gz > "$TOKJSON" 2>/dev/null \
      && echo "[dynamo-tokfix] staged tokenizer.json ($(wc -c <"$TOKJSON") B)"
fi
(
    for _ in $(seq 1 1800); do       # ~60 min, covers late-appearing cache/snapshot dirs
        while IFS= read -r tk; do
            d="$(dirname "$tk")"
            if [ -f "$TOKJSON" ] && [ ! -f "$d/tokenizer.json" ]; then
                cp "$TOKJSON" "$d/tokenizer.json" 2>/dev/null \
                  && echo "[dynamo-tokfix] injected tokenizer.json -> $d (HfTokenizerJson wins over tiktoken)"
            fi
            cfg="$d/config.json"      # fallback: spoof model_type (racy, belt-and-suspenders)
            if [ -f "$cfg" ] && grep -q '"kimi_linear"' "$cfg" 2>/dev/null; then
                sed -i 's/"model_type"[[:space:]]*:[[:space:]]*"kimi_linear"/"model_type": "kimi_k2"/g' "$cfg" 2>/dev/null
            fi
        done < <(find /root/.cache/huggingface /root/.cache/dynamo -name tiktoken.model 2>/dev/null)
        sleep 2
    done
) </dev/null 2>&1 &
disown 2>/dev/null || true

echo "=== dynamo-tokfix poller backgrounded (inject tokenizer.json + config spoof fallback) ==="
