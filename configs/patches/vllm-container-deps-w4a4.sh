#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# W4A4 MegaMoE for DeepSeek-V4-Pro on vLLM (Inferact patch, by zyongye).
#
# Runs in the running vLLM container BEFORE the server launches. Instead of
# building a fresh image (the cu13/aarch64 source build is unfinished in AIB),
# this patches the already-working cu13 container in place:
#   1) Cherry-pick the 3 W4A4 commits' vLLM python changes onto the installed
#      vLLM (additive: kernel.py / envs.py / eplb_utils.py / deepseek_v4 model
#      + prepare_megamoe.py).
#   2) Replace the container's DeepGEMM with the zyongye fork @ 65b3085 (carries
#      the MXFP4 mega kernel; kernels are JIT-compiled at warmup so the wheel
#      build is light).
#
# Enable at runtime with:  --moe-backend deep_gemm_amxf4_mega_moe
# Optional FP8 combine:     VLLM_DSV4_MEGA_FP8_COMBINE=1
#
# Branch: https://github.com/zyongye/vllm/tree/dsv4-megamoe-mxfp4-acts
set -euo pipefail
set -x

# --- base container deps (mirror of vllm-container-deps.sh) ---
apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl git patch
pip install msgpack
if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

VLLM_W4A4_BRANCH="dsv4-megamoe-mxfp4-acts"
DEEPGEMM_REF="65b308581564433f893d416cae09b02b9e64ec16"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# === 1) patch the installed vLLM with the W4A4 commits ===
git clone --depth 60 --branch "$VLLM_W4A4_BRANCH" https://github.com/zyongye/vllm.git "$WORK/vllm"
SP="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
echo "Installed vLLM site-packages: $SP"
# 3 W4A4 commits = 44225c9 (mxfp4 acts) .. e7a908a (dg ref) .. b824873 (fp8 combine)
( cd "$WORK/vllm" && git diff 44225c9^ b824873 -- vllm/ ) > "$WORK/w4a4_vllm.patch"
wc -l "$WORK/w4a4_vllm.patch"
if ( cd "$SP" && git apply -p1 --check "$WORK/w4a4_vllm.patch" ) 2>/dev/null; then
    ( cd "$SP" && git apply -p1 --whitespace=nowarn "$WORK/w4a4_vllm.patch" )
    echo "W4A4 vLLM patch applied cleanly via git apply"
else
    echo "git apply --check failed; falling back to fuzzy patch"
    ( cd "$SP" && patch -p1 --forward --fuzz=3 < "$WORK/w4a4_vllm.patch" )
    echo "W4A4 vLLM patch applied via fuzzy patch"
fi
# fail loud if the new backend literal did not land
python3 -c "from vllm.config.kernel import MoEBackend; import typing; assert 'deep_gemm_amxf4_mega_moe' in typing.get_args(MoEBackend), 'amxf4 backend not registered'" \
    && echo "verified: deep_gemm_amxf4_mega_moe registered"

# === 2) swap DeepGEMM -> zyongye fork @ 65b3085 ===
git clone --recursive --shallow-submodules https://github.com/zyongye/DeepGEMM.git "$WORK/deepgemm"
( cd "$WORK/deepgemm" \
    && git checkout "$DEEPGEMM_REF" \
    && git submodule update --init --recursive --depth 1 \
    && rm -rf build dist ./*.egg-info \
    && python3 setup.py bdist_wheel \
    && python3 -m pip install --force-reinstall --no-deps dist/*.whl )
python3 -c "import deep_gemm; print('deep_gemm from', deep_gemm.__file__)"

echo "W4A4 setup complete: vLLM patched + DeepGEMM=zyongye@${DEEPGEMM_REF}"
