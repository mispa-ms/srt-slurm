#!/bin/bash
# Apply the standard Kimi-K3 container setup, then remove the DS-layout
# speculative-decoding guard so prefix-cache align mode can be exercised.

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh

target=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
patch_file=/configs/patches/remove-vllm-ds-layout-spec-assert.patch
assertion="DS conv state layout does not support mamba align state"

if grep -Fq "${assertion}" "${target}"; then
    patch --batch --forward -p1 -d / < "${patch_file}"
elif grep -Fq "self._mamba_ctx = MambaSpecDecodeGPUContext.create(" "${target}"; then
    echo "DS-layout speculative-decoding assertion is already absent"
else
    echo "ERROR: unexpected vLLM mamba_hybrid.py; refusing to patch" >&2
    exit 1
fi

if grep -Fq "${assertion}" "${target}"; then
    echo "ERROR: DS-layout speculative-decoding assertion is still present" >&2
    exit 1
fi

