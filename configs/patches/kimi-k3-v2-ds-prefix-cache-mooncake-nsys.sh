#!/usr/bin/env bash
# The disagg Mooncake setup plus the Nsight Systems CLI.
# =============================================================================
# kimi-k3-v2-ds-prefix-cache-mooncake.sh with one addition: nsys.
#
# vllm/vllm-openai ships no nsys, and srtctl prefixes the worker launch with
# `nsys profile ...` whenever profiling.type is set. Without the CLI the launch
# exits 127 with "nsys: command not found" and the job dies during bring-up, so
# a profiling config on the plain Mooncake script cannot work at all.
#
# The install block is the same newest-first probe as
# vllm-container-deps-k3-nsys.sh, which is the AGG-side script this mirrors. It
# fails loudly here rather than 127 lines later inside the worker.
#
# Everything above it is unchanged from the Mooncake script: the CUDA 13
# transfer engine, SA's V2 DS prefix-cache backport (vLLM #49291 / #50153) and
# their hybrid KV-load recovery.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl

python3 -m pip install msgpack
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 || true
python3 -m pip install --no-deps 'mooncake-transfer-engine-cuda13==0.3.12.post1'

python3 /configs/patches/patch_kimi_k3_v2_ds_prefix_cache.py
python3 /configs/patches/patch_kimi_k3_mooncake_hma_recompute.py

# The image already has the NVIDIA cuda repo registered — install directly.
# Do NOT add cuda-keyring on top: it triggers an apt Signed-By conflict.
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    for pkg in nsight-systems-2025.6.3 nsight-systems-2025.3.1 nsight-systems nsight-systems-cli; do
        if apt-get install -y --no-install-recommends "$pkg"; then
            echo "[k3-mooncake-nsys] installed $pkg"
            break
        fi
        echo "[k3-mooncake-nsys] package $pkg unavailable, trying next"
    done
fi

command -v nsys >/dev/null 2>&1 || {
    echo "[k3-mooncake-nsys] FATAL - nsys still not on PATH after install attempts" >&2
    exit 1
}
nsys --version
echo "=== k3-mooncake-nsys: done ==="
