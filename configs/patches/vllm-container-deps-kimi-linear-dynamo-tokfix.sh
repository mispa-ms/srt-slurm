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

# Robust persistent poller: dynamo creates >1 mdc tokenizer cache dir (each with its own
# tiktoken.model + config.json); a single early patch missed the 2nd dir and registration
# re-failed. So: patch the config.json next to EVERY tiktoken.model under the whole dynamo
# cache root, every 3s for the whole run, and echo to stdout (frontend log) for visibility.
(
    for _ in $(seq 1 1200); do       # ~60 min, covers late-appearing cache dirs
        while IFS= read -r tk; do
            cfg="$(dirname "$tk")/config.json"
            if [ -f "$cfg" ] && grep -q '"kimi_linear"' "$cfg" 2>/dev/null; then
                sed -i 's/"model_type"[[:space:]]*:[[:space:]]*"kimi_linear"/"model_type": "kimi_k2"/g' "$cfg"
                echo "[dynamo-tokfix] patched $cfg (model_type kimi_linear -> kimi_k2)"
            fi
        done < <(find /root/.cache/dynamo -name tiktoken.model 2>/dev/null)
        sleep 3
    done
) </dev/null 2>&1 &
disown 2>/dev/null || true

echo "=== dynamo-tokfix poller backgrounded (patches config.json by every tiktoken.model) ==="
