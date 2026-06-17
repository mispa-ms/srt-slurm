#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Mooncake CPU KV-offload setup for vLLM agg (AgentX).
# Mirrors InferenceX benchmarks/single_node/agentic/dsv4_fp4_b300_vllm.sh `cpu` case:
# installs the mooncake store, writes its config, and launches a local mooncake_master.
# The recipe pairs this with:
#   backend.vllm_config.aggregated.kv-transfer-config:
#     '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
#   backend.aggregated_environment: MOONCAKE_CONFIG_PATH=/tmp/mooncake_config.json + MC_* knobs (see recipe).

set -euo pipefail

# Base container deps first (numactl, msgpack, numa-bind fix).
bash /configs/patches/vllm-container-deps.sh

# ---- Mooncake store install (cu13 wheel for B300/GB300) ----------------------
MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.11.post1}"
pip install --quiet --no-cache-dir --no-deps --force-reinstall \
    "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
echo "[mooncake] installed ${MOONCAKE_VERSION}"

# ---- Mooncake config -----------------------------------------------------------
# bia B300 host RAM = 2014 GiB; use a 1500 GB aggregate CPU pool (~500 GiB headroom).
# Mooncake embedded mode: each of TP ranks contributes one global segment to the
# shared local store, so pre-divide by TP. TP8 -> 1500/8 = 187 GB/rank.
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1500}"
TP="${MOONCAKE_TP:-8}"
PER_RANK_GB=$(( TOTAL_CPU_DRAM_GB / TP ))
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-20888}"   # high free port; 8700 collides with srt-slurm's reserved mooncake port (admin server bind fails)
MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-/tmp/mooncake_config.json}"
MOONCAKE_MASTER_LOG="${MOONCAKE_MASTER_LOG:-/tmp/mooncake_master.log}"

cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:${MOONCAKE_MASTER_PORT}",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
EOF
echo "[mooncake] wrote $MOONCAKE_CONFIG_PATH (per-rank ${PER_RANK_GB}GB x TP${TP})"

# ---- Launch local mooncake_master (detached so it survives this script) -------
echo "[mooncake] starting master on 127.0.0.1:${MOONCAKE_MASTER_PORT}"
# NOTE: the master's admin/metrics HTTP server fails to bind even on a free high port
# (29003) in this env -> "Failed to start master admin server" -> exit 11. It's not a
# port conflict (the bind itself fails); we don't need master metrics for the benchmark,
# so DISABLE it. RPC service (client connects here, --port) starts fine independently.
setsid nohup mooncake_master --port "${MOONCAKE_MASTER_PORT}" \
    --enable_metric_reporting=false \
    --eviction_high_watermark_ratio=0.80 \
    --eviction_ratio=0.10 \
    > "${MOONCAKE_MASTER_LOG}" 2>&1 &
MOONCAKE_MASTER_PID=$!
disown "${MOONCAKE_MASTER_PID}" 2>/dev/null || true
sleep 3
if ! kill -0 "${MOONCAKE_MASTER_PID}" 2>/dev/null; then
    echo "[mooncake] master died during startup:" >&2
    cat "${MOONCAKE_MASTER_LOG}" >&2
    exit 1
fi
echo "[mooncake] master up (pid ${MOONCAKE_MASTER_PID})"
