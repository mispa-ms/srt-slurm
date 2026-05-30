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

# Nsight Systems CLI. The Ubuntu 24.04 default repo (and NVIDIA cuda repo's
# `nsight-systems-cli` package) only carries nsys 2024.2.3, which lacks
# `--gpu-metrics-devices=cuda-visible` (added in nsys 2024.5+). srtctl injects
# that flag into worker launches, so an older nsys exits unrecognised-option.
#
# Fix: install the full `nsight-systems-2025.6.3` package from NVIDIA cuda repo
# — that variant goes up to 2026.x while `-cli` is stuck at 2024.2.3.
# Confirmed by inspecting the repo Packages index for ubuntu2404/sbsa.
if ! command -v nsys >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends wget ca-certificates
    ARCH=$(dpkg --print-architecture)
    if [ "$ARCH" = "arm64" ]; then KEYRING_ARCH=sbsa; else KEYRING_ARCH=x86_64; fi
    wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${KEYRING_ARCH}/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb
    apt-get -y update
    apt-get install -y --no-install-recommends nsight-systems-2025.6.3
    rm -f /tmp/cuda-keyring.deb
fi
nsys --version
