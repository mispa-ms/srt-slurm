#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# INSTALL-ONLY mooncake setup for vLLM DISAGG (P/D) — installs the mooncake python
# package (arch/glibc-aware) but does NOT write a config or launch a master. Use this
# when the srt-slurm native `mooncake_kv_store` block handles the SHARED master +
# MOONCAKE_MASTER injection + MOONCAKE_CONFIG_PATH JSON render (the disagg case). The
# embedded-master variant (vllm-container-deps-mooncake.sh) is for AGG single-node and
# would conflict with the shared master in disagg.
#
# Needed because stock vLLM nightly images do NOT ship the mooncake pkg, so a config
# using MooncakeStoreConnector fails at import: "Please install mooncake ...".

set -euo pipefail

# Base container deps first (numactl, msgpack, numa-bind fix + any gated PR patches).
bash /configs/patches/vllm-container-deps.sh

# ---- Mooncake store install (arch/glibc-aware; cu13 aarch64 needs care) ---------
# The cuda13 aarch64 wheel is manylinux_2_39 (glibc>=2.39, Ubuntu 24.04). The vLLM
# nightly arm64 container is 22.04 (glibc 2.35), so on GB300 pip can't install it —
# fall back to the non-cuda13 manylinux_2_35 wheel (0.3.9 >= vLLM kv_connectors floor
# 0.3.8) and provide libcudart.so.12 via the cu12 runtime shim so it IMPORTS on cu13.
#   - x86 (bia): cuda13 wheel (manylinux_2_35, glibc 2.35 ok).
#   - aarch64 + cu13 + glibc<2.39: non-cuda13 wheel + cudart12 shim.
#   - aarch64 + cu12: non-cuda13 wheel.
MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.11.post1}"
CU_MAJOR=$(ldconfig -p 2>/dev/null | grep -oE 'libcudart\.so\.[0-9]+' | grep -oE '[0-9]+$' | sort -un | tail -1)
GLIBC_MINOR=$(getconf GNU_LIBC_VERSION 2>/dev/null | grep -oE '[0-9]+$')
echo "[mooncake-install] arch=$(uname -m) cuda_major=${CU_MAJOR:-unknown} glibc=2.${GLIBC_MINOR:-?}"
NEED_CUDART12_SHIM=0
if [ "$(uname -m)" = "aarch64" ] && [ "${CU_MAJOR}" = "12" ]; then
    MOONCAKE_PKG="mooncake-transfer-engine==${MOONCAKE_AARCH_VERSION:-0.3.9}"
elif [ "$(uname -m)" = "aarch64" ] && [ "${CU_MAJOR}" = "13" ] && [ -n "${GLIBC_MINOR}" ] && [ "${GLIBC_MINOR}" -lt 39 ]; then
    MOONCAKE_PKG="mooncake-transfer-engine==${MOONCAKE_AARCH_VERSION:-0.3.9}"
    NEED_CUDART12_SHIM=1
else
    MOONCAKE_PKG="mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
fi
if [ "${NEED_CUDART12_SHIM}" = "1" ]; then
    echo "[mooncake-install] cu13 image + glibc<2.39 -> cu12 runtime shim (libcudart.so.12)"
    pip install --quiet --no-cache-dir "nvidia-cuda-runtime-cu12"
    CUDART12_SO=""
    LOC=$(pip show nvidia-cuda-runtime-cu12 2>/dev/null | awk -F': ' '/^Location:/{print $2}')
    [ -n "${LOC}" ] && CUDART12_SO=$(ls "${LOC}"/nvidia/cuda_runtime/lib/libcudart.so.12* 2>/dev/null | head -1)
    [ -z "${CUDART12_SO}" ] && CUDART12_SO=$(find / -maxdepth 9 -path '*cuda_runtime*' -name 'libcudart.so.12*' 2>/dev/null | head -1)
    if [ -n "${CUDART12_SO}" ] && [ -f "${CUDART12_SO}" ]; then
        cp -f "${CUDART12_SO}" /usr/local/lib/ 2>/dev/null || cp -f "${CUDART12_SO}" /usr/lib/
        ldconfig 2>/dev/null || true
        echo "[mooncake-install] installed libcudart.so.12 shim: $(basename "${CUDART12_SO}")"
    else
        echo "[mooncake-install] WARN: could not locate cu12 libcudart.so.12 — import may fail"
    fi
fi
echo "[mooncake-install] installing ${MOONCAKE_PKG}"
pip install --quiet --no-cache-dir --no-deps --force-reinstall "${MOONCAKE_PKG}"
python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
echo "[mooncake-install] installed ${MOONCAKE_PKG} — import OK"

# Bound MooncakeStoreConnector transfer batches (InferenceX patch): mooncake's TCP
# connection pool grows unbounded, so big agentic per-layer transfers exhaust the
# node's ephemeral ports. Splits batch_put/get into chunks. Idempotent.
if [ -f /configs/patches/patch_vllm_mooncake_transfer_batches.py ]; then
    python3 /configs/patches/patch_vllm_mooncake_transfer_batches.py \
        && echo "[mooncake-install] transfer-batching patch applied" \
        || echo "[mooncake-install] WARN: transfer-batching patch failed (worker.py anchors may differ)"
fi

echo "[mooncake-install] done (no master launched — srt-slurm mooncake_kv_store block owns the shared master)"
