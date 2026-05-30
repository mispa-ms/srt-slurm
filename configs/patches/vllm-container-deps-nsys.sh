#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same as vllm-container-deps.sh, plus Nsight Systems CLI for profiling-enabled
# vLLM recipes (e.g. disagg-1p1d-dep4-dep16-decode-only-c4096-nsys.yml).
# The base vllm/vllm-openai images don't ship nsys, so worker launches that
# wrap with `nsys profile ...` exit 127.

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# Nsight Systems CLI. Install nsight-systems-cli (nsys 2024.2.3 in NVIDIA's
# ubuntu2404/sbsa cuda repo). Older but reliable to install: the
# `nsight-systems-<version>` family triggers an apt Signed-By conflict when
# combined with the cuda-keyring deb on top of vllm/vllm-openai images that
# already have a cuda repo registered.
#
# Note: 2024.2.3 doesn't support `--gpu-metrics-devices=cuda-visible` (added
# in nsys 2024.5+). YAMLs that need that flag should drop it from
# `profiling.extra_nsys_args` — kernel + CUDA API + NVTX traces still work.
if ! command -v nsys >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends wget ca-certificates
    ARCH=$(dpkg --print-architecture)
    if [ "$ARCH" = "arm64" ]; then KEYRING_ARCH=sbsa; else KEYRING_ARCH=x86_64; fi
    wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${KEYRING_ARCH}/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb
    apt-get -y update
    apt-get install -y --no-install-recommends nsight-systems-cli
    rm -f /tmp/cuda-keyring.deb
fi
nsys --version
