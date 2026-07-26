#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# vllm-container-deps.sh plus the Nsight Systems CLI, for recipes that set
# `profiling.type: nsys-time`. The vllm/vllm-openai images don't ship nsys, so
# the worker launch (which srtctl prefixes with `nsys profile ...`) exits 127
# with "nsys: command not found".
#
# Time-based capture only — no CudaProfilerWrapper patching. nsys-time drives
# the window with --delay/--duration, so none of the cudaProfilerApi
# start/stop coordination is involved.

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# The image already has the NVIDIA cuda repo registered — install directly.
# Do NOT add cuda-keyring on top: it triggers an apt Signed-By conflict.
#
# Package availability differs per distro/arch, so try newest-first and stop at
# the first hit. `nsight-systems-cli` is the fallback (older, but enough for
# --delay/--duration capture).
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    for pkg in nsight-systems-2025.6.3 nsight-systems-2025.3.1 nsight-systems nsight-systems-cli; do
        if apt-get install -y --no-install-recommends "$pkg"; then
            echo "installed $pkg"
            break
        fi
        echo "package $pkg unavailable, trying next"
    done
fi

# Fail loudly here rather than 127 lines later inside the worker launch.
command -v nsys >/dev/null 2>&1 || { echo "FATAL: nsys still not on PATH after install attempts" >&2; exit 1; }
nsys --version
