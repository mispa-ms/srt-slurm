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

# ---- PR #48180 (DCP + Eagle for Tokenspeed_MLA / FlashInfer) runtime patch — gated on APPLY_PR48180 ----
# 4-file python patch on top of the latest nightly + tokenspeed-mla 0.1.2->0.1.8. Enables DCP4 + eagle3
# spec-decode with TOKENSPEED_MLA attention (Pavani #48180, GSM8k cudagraph-verified). Unmerged/open.
if [ "${APPLY_PR48180:-0}" = "1" ] && [ -f /configs/patches/pr48180-on-latest.patch ]; then
    echo "[pr48180] installing tokenspeed-mla==0.1.8"
    pip install "tokenspeed-mla==0.1.8" || { echo "[pr48180] tokenspeed-mla install FAILED"; exit 1; }
    VLLM_SITE=$(python3 -c "import os,vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))")
    echo "[pr48180] applying DCP+Eagle Tokenspeed patch to $VLLM_SITE"
    ( cd "$VLLM_SITE" && git apply -p1 -v /configs/patches/pr48180-on-latest.patch ) \
      || ( cd "$VLLM_SITE" && patch -p1 --forward --batch < /configs/patches/pr48180-on-latest.patch ) \
      || { echo "[pr48180] PATCH FAILED"; exit 1; }
    echo "[pr48180] patch applied OK"
fi

# ---- PR #48248 (FlashInfer fused MNNVL A2A for DCP a2a-reduce, Blackwell auto-select) — gated on APPLY_PR48248 ----
# Adds vllm/distributed/dcp_alltoall_flashinfer.py + wires it into dcp_alltoall.py / gpu_worker.py. On sm_100+
# (GB300) the fused LL128 decode_cp_a2a_alltoall replaces NCCL pack->all_to_all->unpack under CUDA graph. Unmerged/open.
if [ "${APPLY_PR48248:-0}" = "1" ] && [ -f /configs/patches/pr48248-on-latest.patch ]; then
    VLLM_SITE=$(python3 -c "import os,vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))")
    echo "[pr48248] applying FlashInfer fused DCP-A2A patch to $VLLM_SITE"
    ( cd "$VLLM_SITE" && git apply -p1 -v /configs/patches/pr48248-on-latest.patch ) \
      || ( cd "$VLLM_SITE" && patch -p1 --forward --batch < /configs/patches/pr48248-on-latest.patch ) \
      || { echo "[pr48248] PATCH FAILED"; exit 1; }
    echo "[pr48248] patch applied OK"
fi

# ---- NIXL decode-only draft-layer exclusion (EAGLE3/Kimi decode-only P/D) — gated on APPLY_NIXL_DECODE_ONLY ----
# Excludes decode-local draft (eagle3/spec) KV layers from the NIXL P/D transfer set, so spec can be enabled
# DECODE-ONLY without the "Number of KV layers must match" handshake assert. Covers V1(EAGLE3/Kimi) + V2(DSpark)
# + MultiConnector. Pure-Python (7 files), applies clean on nightly-2c17d33f. Branch misunp/nixl-eagle-decode-only-pd.
if [ "${APPLY_NIXL_DECODE_ONLY:-0}" = "1" ] && [ -f /configs/patches/nixl-decode-only-pd.patch ]; then
    VLLM_SITE=$(python3 -c "import os,vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))")
    echo "[nixl-dco] applying NIXL decode-only draft-layer exclusion patch to $VLLM_SITE"
    ( cd "$VLLM_SITE" && git apply -p1 -v /configs/patches/nixl-decode-only-pd.patch ) \
      || ( cd "$VLLM_SITE" && patch -p1 --forward --batch < /configs/patches/nixl-decode-only-pd.patch ) \
      || { echo "[nixl-dco] PATCH FAILED"; exit 1; }
    echo "[nixl-dco] patch applied OK"
fi

# ---- DCP profiling-assert fix (PR #40996 regression) — always applied ----------
# PR #40996 added prepare_dcp_dummy_context_metadata() which asserts
# kv_cache_config.num_blocks >= 2, but determine_available_memory profiles the
# cudagraph with a MINIMAL placeholder KV config (num_blocks <= 1). On disagg
# decode-only workers (uniform_decode) this fires and crashes cudagraph capture
# ("assert max_valid_block_id > 0"). AGG workers are mixed prefill+decode so
# get_dcp_dummy_context_len() returns 0 there and never hits the assert — which
# is why AGG DCP4 works and disagg DCP4 dies. Patch skips the dummy-context fill
# when num_blocks <= 1 (matches pre-#40996 behavior). Idempotent; non-fatal on
# pre-#40996 images (anchor absent). Run AFTER the git-apply PR patches so it
# operates on the final cp_utils.py.
if [ -f /configs/patches/vllm_dcp_profiling_assert_fix.py ]; then
    python3 /configs/patches/vllm_dcp_profiling_assert_fix.py
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
