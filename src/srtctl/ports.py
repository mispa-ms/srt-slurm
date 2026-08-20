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

# Mooncake transfer-engine ports (shared by SGLang and vLLM backends).
MOONCAKE_MASTER_PORT = 8700
MOONCAKE_HTTP_METADATA_PORT = 8701
# Master's admin HTTP server (Prometheus metrics + /health, /role, /query_key, …).
# Mooncake's compile-time default is 9003; we pass --metrics_port explicitly so
# the master lives entirely inside our consolidated 8700-range.
MOONCAKE_METRICS_PORT = 8702

# vLLM backend ports.
VLLM_NIXL_PORT_BASE = 5400
VLLM_DATA_PARALLEL_RPC_PORT = 8400
# vLLM's multi-node rendezvous port. Left unset, vLLM defaults to 29500(+1) and
# a stale bind from any earlier job on the node kills the run with
# "server socket has failed to listen ... port: 29501 ... EADDRINUSE".
# Derived per SLURM job so two runs, or a run and someone's leftovers, do not
# share it. 1024 slots is far more than a cluster runs at once.
VLLM_MASTER_PORT_BASE = 26000
VLLM_MASTER_PORT_SLOTS = 1024

VLLM_PORT_BASE = 20000
VLLM_PORT_STRIDE = 50

# Dynamo runtime and connector ports.
DYN_SYSTEM_PORT_BASE = 7500
KVBM_ZMQ_PORT_BASE = 5600
