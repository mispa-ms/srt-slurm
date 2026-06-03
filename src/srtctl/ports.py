# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized default ports used by srt-slurm runtime components."""

# Shared infrastructure services.
ETCD_CLIENT_PORT = 2379
ETCD_PEER_PORT = 2380
NATS_PORT = 4222

# Frontend service ports.
FRONTEND_PUBLIC_PORT = 8000
FRONTEND_INTERNAL_PORT = 8180

# Shared worker endpoint ports.
# SGLang uses this for --kv-events-config; vLLM uses it for
# DYN_VLLM_KV_EVENT_PORT.
KV_EVENTS_PORT_BASE = 5200

# SGLang backend ports.
SGLANG_HTTP_PORT_BASE = 6100
SGLANG_HTTP_PORT_STRIDE = 32
SGLANG_BOOTSTRAP_PORT_BASE = 7200
SGLANG_DIST_INIT_PORT_BASE = 8300

# SGLang Mooncake transfer-engine ports.
MOONCAKE_MASTER_PORT = 8700
MOONCAKE_HTTP_METADATA_PORT = 8701

# vLLM backend ports.
VLLM_NIXL_PORT_BASE = 5400
VLLM_DATA_PARALLEL_RPC_PORT = 8400

# vLLM NIXL side-channel stride: the handshake listener binds at
#   VLLM_NIXL_SIDE_CHANNEL_PORT + STRIDE * rank
# where rank is dp_rank (DP-EP mode) or tp_rank (TP-only mode). The block
# reserved per replica must be `size * STRIDE` to prevent adjacent replicas
# from colliding on the odd-stride ports of the previous block.
#
# Known stride values by vLLM version:
#   - vLLM 0.22.x: STRIDE = 2 (verified from prefill worker logs:
#       EngineCore_DP0 @ base+0, EngineCore_DP1 @ base+2, ...)
#   - vLLM 0.21.x and earlier: STRIDE = 1 inferred (the older
#       `actual_port = base + rank` documentation implies contiguous
#       per-rank ports; users of 0.21.x should explicitly override
#       VLLMProtocol.nixl_port_stride=1 to be safe).
# Users can override per-recipe via VLLMProtocol.nixl_port_stride (set under
# the backend block in the recipe YAML).
VLLM_NIXL_PORT_STRIDE_DEFAULT = 2

# Dynamo runtime and connector ports.
DYN_SYSTEM_PORT_BASE = 7500
KVBM_ZMQ_PORT_BASE = 5600
