#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# ---- PR #38433 (nixl DCP for PD-disagg, unequal dcp) runtime patch — gated on APPLY_PR38433 ----
# Python-only 10-file patch, cherry-picked onto cbe9c40f. Fixes tp-only KV block mapping
# (DEP4->DCP correctness). Requires cp_kv_cache_interleave_size == block_size in the vLLM config.
if [ "${APPLY_PR38433:-0}" = "1" ] && [ -f /configs/patches/pr38433-on-cbe9c40f.patch ]; then
    VLLM_SITE=$(python3 -c "import os,vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))")
    echo "[pr38433] applying nixl-DCP patch to $VLLM_SITE"
    ( cd "$VLLM_SITE" && git apply -p1 -v /configs/patches/pr38433-on-cbe9c40f.patch ) \
      || ( cd "$VLLM_SITE" && patch -p1 --forward --batch < /configs/patches/pr38433-on-cbe9c40f.patch ) \
      || { echo "[pr38433] PATCH FAILED"; exit 1; }
    echo "[pr38433] patch applied OK"
fi

# ---- PR #45340 (CP-scaled scheduler block accounting, aligned DCP) — gated on APPLY_PR45340 ----
# Python-only 3-file patch (nixl base_scheduler + metadata, mooncake connector), cherry-picked
# onto cbe9c40f. Fixes aligned-DCP PD-disagg: connector schedulers must scale block_size by
# dcp_size*pcp_size (each block covers that many interleaved tokens) — raw block_size over-counts
# and hangs long-context prefill KV transfer. Needed for 1P1D dcp4-dcp4 with NIXL/Mooncake.
if [ "${APPLY_PR45340:-0}" = "1" ] && [ -f /configs/patches/pr45340-on-cbe9c40f.patch ]; then
    VLLM_SITE=$(python3 -c "import os,vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))")
    echo "[pr45340] applying aligned-DCP block-accounting patch to $VLLM_SITE"
    ( cd "$VLLM_SITE" && git apply -p1 -v /configs/patches/pr45340-on-cbe9c40f.patch ) \
      || ( cd "$VLLM_SITE" && patch -p1 --forward --batch < /configs/patches/pr45340-on-cbe9c40f.patch ) \
      || { echo "[pr45340] PATCH FAILED"; exit 1; }
    echo "[pr45340] patch applied OK"
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
