#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# ---- GPU state at container setup (diagnose startup free-memory OOM) ----------
# Dump per-GPU memory + any resident compute processes BEFORE vLLM allocates, so
# a "free memory < gpu-memory-utilization" startup failure reveals WHO holds the
# memory (stale/zombie process on a dirty node vs a clean GPU). Non-fatal.
echo "[gpu-diag] nvidia-smi at setup on $(hostname) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}):"
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv 2>&1 || true
echo "[gpu-diag] resident compute processes (who holds GPU memory):"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>&1 || true

# ---- host-mem poller (KV-offload pool sizing visibility) ----------------------
# Background logger: every 30s prints total/used/avail + a RUNNING PEAK to stdout
# (inherited by the worker log, so it lands in the run artifact). Lets us confirm
# whether the pinned CPU KV-offload pool actually fills and how close we run to
# the node RAM ceiling — the runtime high-water-mark the agentx bench doesn't log.
(
  set +e
  peak=0
  while :; do
    tot=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    used=$(( (tot - avail) / 1048576 ))            # GiB
    [ "$used" -gt "$peak" ] && peak=$used
    echo "[hostmem] $(date -u +%H:%M:%S) used=${used}GiB avail=$(( avail / 1048576 ))GiB peak=${peak}GiB total=$(( tot / 1048576 ))GiB"
    sleep 30
  done
) &
disown $! 2>/dev/null || true
echo "[hostmem] poller started (pid $!), logging every 30s to the worker stdout"
