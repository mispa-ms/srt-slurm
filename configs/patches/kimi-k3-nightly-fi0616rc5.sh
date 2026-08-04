#!/bin/bash
# Kimi-K3 on the vLLM NIGHTLY image, with FlashInfer pinned to v0.6.16rc5.
#
# Everything we have measured so far ran on vllm/vllm-openai:kimi-k3 — a
# hand-assembled image whose FlashInfer does not even satisfy its own vLLM:
#
#   vllm 0.1.dev19262+gb6bbf29dd.d20260727 requires flashinfer-python==0.6.15.post1,
#   but you have flashinfer-python 0.6.15 which is incompatible.
#
# So the version a container actually carries cannot be read off
# requirements/cuda.txt. This script prints what is installed before and after
# it touches anything, and the first log tells us what nightly really ships.
#
# FlashInfer v0.6.16rc5 is the version the team recorded as the one to install
# for K3 on nightly. It is also the first release carrying the MegaMoE kernels,
# which is the next thing on the list after this.
#
# Three packages have to move together or the JIT path mixes versions:
#   flashinfer-python     PyPI
#   flashinfer-cubin      flashinfer.ai/whl (not on PyPI since 0.6.14)
#   flashinfer-jit-cache  flashinfer.ai/whl/cu130 (CUDA-specific build)

set -euo pipefail

FI_VER="${FI_VER:-0.6.16rc5}"
FI_CUDA="${FI_CUDA:-cu130}"

echo "=== flashinfer/vllm BEFORE ==="
python3 -m pip list 2>/dev/null | grep -iE "flashinfer|^vllm|^torch |nccl" || true

bash /configs/patches/vllm-container-deps.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl

python3 -m pip install msgpack
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 || true
python3 -m pip install --no-deps 'mooncake-transfer-engine-cuda13==0.3.12.post1'

# --no-deps so pip cannot drag vLLM's pinned flashinfer back in; the version
# conflict warning it prints afterwards is expected and is the point.
echo "=== installing flashinfer ${FI_VER} (${FI_CUDA}) ==="
python3 -m pip install --no-deps --force-reinstall \
    "flashinfer-python==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/" \
    "flashinfer-cubin==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/${FI_CUDA}/" \
    "flashinfer-jit-cache==${FI_VER}+${FI_CUDA}" || \
    echo "WARNING: flashinfer-jit-cache ${FI_VER}+${FI_CUDA} not installed; JIT will compile on demand"

echo "=== flashinfer AFTER ==="
python3 -m pip list 2>/dev/null | grep -iE "flashinfer|^vllm" || true
python3 -c "import flashinfer, sys; print('flashinfer import OK:', flashinfer.__version__)" || \
    { echo "FATAL: flashinfer ${FI_VER} does not import"; exit 1; }

# Nightly carries Kimi-K3 natively (vLLM PR #50000, merged 2026-07-30), so the
# kimi-k3 image's model patches are not applied here. The hybrid-KV load-failure
# recompute is still needed — check whether nightly already has it before
# assuming this line is a no-op.
python3 /configs/patches/patch_kimi_k3_mooncake_hma_recompute.py || \
    echo "NOTE: hma_recompute patch did not apply — nightly may already carry it"
