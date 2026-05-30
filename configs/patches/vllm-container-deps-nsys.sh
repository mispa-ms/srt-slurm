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

# Nsight Systems CLI. The vllm/vllm-openai image already has the NVIDIA
# cuda repo registered, so a direct `apt install` works. Installing
# cuda-keyring on top triggers an apt Signed-By conflict against the
# container's pre-registered entry (seen on #53132895, #53135432) — skip it.
#
# Note: 2024.2.3 doesn't support `--gpu-metrics-devices=cuda-visible` (added
# in nsys 2024.5+). YAMLs that need GPU metrics should use the legacy form
# `--gpu-metrics-devices=0` (or `=all`) — each DP rank is wrapped by its own
# nsys with CUDA_VISIBLE_DEVICES pinned to one GPU.
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    apt-get install -y --no-install-recommends nsight-systems-cli
fi
nsys --version
