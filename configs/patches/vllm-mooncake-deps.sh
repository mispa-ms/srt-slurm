#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for vLLM disagg recipes that use MooncakeConnector
# (instead of NixlConnector). Same as vllm-container-deps.sh plus
# mooncake-transfer-engine install.

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack
pip install 'mooncake-transfer-engine>=0.3.8'

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi
