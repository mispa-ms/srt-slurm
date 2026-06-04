#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same as vllm-container-deps.sh, plus Nsight Systems CLI for profiling-enabled
# vLLM recipes. The base vllm/vllm-openai images don't ship nsys, so worker
# launches that wrap with `nsys profile ...` exit 127.

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# Nsight Systems. The vllm/vllm-openai image already has the NVIDIA cuda repo
# registered, so a direct `apt install` works. Installing cuda-keyring on top
# triggers an apt Signed-By conflict.
#
# `nsight-systems-cli` is pinned to 2024.2.3 on ubuntu2404/sbsa, which lacks
# `--gpu-metrics-devices`. The full `nsight-systems-2025.6.3` package carries
# the version we need (confirmed via the repo Packages index for
# ubuntu2404/sbsa).
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    apt-get install -y --no-install-recommends nsight-systems-2025.6.3
fi
nsys --version
