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
MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-/tmp/mooncake_config.json}"
MOONCAKE_MASTER_LOG="${MOONCAKE_MASTER_LOG:-/tmp/mooncake_master.log}"

# ---- Port selection (ROOT CAUSE of prior failures) ----------------------------
# mooncake_master starts TWO servers: a coro_rpc server on --port and a MANDATORY
# cinatra admin/metrics HTTP server on --metrics_port (master.cpp:1100-1105 →
# `return 1` if it can't bind; rpc_service.cpp:228-234; not gated by
# enable_metric_reporting). Both bind 0.0.0.0 via identical code. On bia the RPC
# port (20888) was free but the metrics port (default 9003 AND a hard-coded 29003)
# was already LISTENing in the node's network namespace, so RPC came up and the
# admin bind failed fatally. The InferenceX dsv4 reference omits --metrics_port
# (uses 9003) and only works because its cluster has 9003 free. Fix: probe a
# genuinely-free port for each server instead of guessing, and log what holds the
# old ports so the occupancy is provable in the run log.
_dump_listeners() {
    python3 - <<'PY' || true
import glob
def ports(path):
    out=set()
    try:
        for ln in open(path).read().splitlines()[1:]:
            f=ln.split()
            if len(f)>3 and f[3]=='0A':  # 0A = TCP LISTEN
                out.add(int(f[1].rsplit(':',1)[1],16))
    except FileNotFoundError:
        pass
    return out
listening=ports('/proc/net/tcp')|ports('/proc/net/tcp6')
for p in (9003,20888,29003):
    print(f"[mooncake]   port {p}: {'IN USE' if p in listening else 'free'}")
print(f"[mooncake]   ({len(listening)} TCP listeners in this netns)")
PY
}
_free_port() {
    # Ask the kernel for an ephemeral free port, bound to 0.0.0.0 exactly like
    # mooncake_master does, so the result is representative.
    python3 -c 'import socket;s=socket.socket();s.bind(("0.0.0.0",0));print(s.getsockname()[1]);s.close()'
}
echo "[mooncake] listener check before port selection:"
_dump_listeners
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-$(_free_port)}"     # coro_rpc server
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-$(_free_port)}"   # mandatory cinatra admin/metrics server
echo "[mooncake] selected rpc=${MOONCAKE_MASTER_PORT} metrics=${MOONCAKE_METRICS_PORT}"

cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:${MOONCAKE_MASTER_PORT}",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "4GB",
  "protocol": "tcp",
  "device_name": "",
  "enable_offload": false
}
EOF
echo "[mooncake] wrote $MOONCAKE_CONFIG_PATH (per-rank ${PER_RANK_GB}GB x TP${TP})"

# ---- Launch local mooncake_master (detached so it survives this script) -------
echo "[mooncake] starting master rpc=127.0.0.1:${MOONCAKE_MASTER_PORT} metrics=${MOONCAKE_METRICS_PORT}"
# The admin/metrics HTTP server is mandatory (master.cpp:1100-1105 → exit on bind
# failure) and cannot be disabled by any flag; enable_metric_reporting only gates
# the logging thread. So both ports must be genuinely free — see the probe above.
setsid nohup mooncake_master --port "${MOONCAKE_MASTER_PORT}" \
    --metrics_port="${MOONCAKE_METRICS_PORT}" \
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
