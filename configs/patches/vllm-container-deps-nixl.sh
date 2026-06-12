#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Base container deps + NIXL (KV-transfer for disagg).
#
# Our self-built W4A4 image (gitlab .../vllm_aarch64:54510032) was built from
# zyongye/vllm's docker/Dockerfile WITHOUT the INSTALL_KV_CONNECTORS=true build
# arg, so NIXL (nixl-cu13, used by NixlConnector for disagg KV transfer) was not
# installed. The dynamo vLLM worker imports dynamo.nixl_connect at startup and
# dies with "No module named 'nixl'". Install the CUDA-13 NIXL wheel here to
# match the Dockerfile's `pip install nixl-cu${CUDA_MAJOR}` line.
set -euo pipefail
set -x

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl
pip install msgpack
if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# NIXL: base `nixl` provides the python module; `nixl-cu13` force-reinstalls
# the matching CUDA-13 native nixl_ep_cpp.so (mirrors the Dockerfile order).
python3 -m pip install nixl
python3 -m pip install --force-reinstall --no-deps nixl-cu13
python3 -c "import nixl._api; print('nixl OK:', nixl._api.__file__)"

echo "vllm-container-deps-nixl: base deps + nixl-cu13 installed"
