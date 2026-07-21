#!/usr/bin/env bash
# Kimi-Linear dynamo tokenizer fix — vendored dynamo-tokenizers v1.5.0 + the kimi_linear one-liner.
# =============================================================================
# WHY: dynamo's frontend tiktoken loader rejects model_type 'kimi_linear' (fixed upstream by
#   ai-dynamo/frontend-crates#134). We can't just build dynamo against that PR branch: the branch
#   is on dynamo-tokenizers 1.5.3, whose API dynamo main (pinned to "=1.5.0") does NOT compile
#   against (error[E0599]: no method `with_extend` in dynamo-llm). The actual fix, though, is a
#   single API-independent line in tiktoken.rs, so we apply it onto the 1.5.0 crate instead.
#
# HOW: clone frontend-crates at the dynamo-tokenizers-v1.5.0 tag, sed in the kimi_linear match arm,
#   and let the dynamo source build override `dynamo-tokenizers -> { path = /tmp/ftz/tokenizers }`
#   (via dynamo.cargo_patches). dynamo main then compiles against a 1.5.0 crate (API match) that
#   accepts kimi_linear. Only the build node needs /tmp/ftz; the resulting wheel is self-contained.
# =============================================================================
set -euo pipefail

echo "=== Kimi-Linear ftz-vendor: base deps + vendored dynamo-tokenizers v1.5.0 + kimi_linear fix ==="

if [[ -f /configs/patches/vllm-container-deps.sh ]]; then
    bash /configs/patches/vllm-container-deps.sh
fi

if ! command -v git >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1 || true
fi

FTZ=/tmp/ftz
TIK="$FTZ/tokenizers/src/tiktoken.rs"
rm -rf "$FTZ"
git clone --depth 1 --branch dynamo-tokenizers-v1.5.0 \
    https://github.com/ai-dynamo/frontend-crates "$FTZ" 2>&1 | tail -1

if [[ -f "$TIK" ]]; then
    sed -i 's/"kimi" | "kimi_k2" | "kimi_k25" | "deepseek_v3" => Ok(KIMI_PATTERN)/"kimi" | "kimi_k2" | "kimi_k25" | "kimi_linear" | "deepseek_v3" => Ok(KIMI_PATTERN)/' "$TIK"
    if grep -q '"kimi_linear"[[:space:]]*|[[:space:]]*"deepseek_v3"' "$TIK"; then
        echo "[ftz-vendor] applied kimi_linear fix to $TIK"
    else
        echo "[ftz-vendor] WARNING: kimi_linear fix NOT applied (match line changed upstream?)" >&2
    fi
else
    echo "[ftz-vendor] WARNING: $TIK not found — clone failed?" >&2
fi

echo "=== ftz-vendor complete ==="
