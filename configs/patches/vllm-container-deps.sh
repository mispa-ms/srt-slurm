#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# ---- vLLM #47589 overlay: FlashInfer MNNVL allreduce one-shot workspace fix ----
# #47589 is a pure-Python, single-file change that fixes the cross-node TP8 crash
# "buffer size in the given workspace is insufficient" during CUDA-graph capture
# (delegates the mnnvl one-shot decision to FlashInfer AUTO). Our image
# nightly-93d8f834 predates it. Overlay the cherry-picked file (base 93d8f834 +
# ONLY #47589) IFF the in-image file is byte-identical to the base we validated
# against (sha256 guard). If a future image already has the fix, or changed this
# file, we skip instead of reverting it. Non-fatal.
PR47589_SRC=/configs/patches/allreduce_rms_fusion.pr47589.py
PR47589_BASE_SHA=6425e1c24c2b25faeec16c8e753148037862f227d7807135519977d89b5807f8
if [ -f "$PR47589_SRC" ]; then
    VLLM_DIR=$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)
    TGT="$VLLM_DIR/compilation/passes/fusion/allreduce_rms_fusion.py"
    if [ -n "$VLLM_DIR" ] && [ -f "$TGT" ]; then
        if grep -q _select_flashinfer_allreduce_use_oneshot "$TGT"; then
            echo "[pr47589] skip: image already contains #47589"
        elif [ "$(sha256sum "$TGT" | awk '{print $1}')" = "$PR47589_BASE_SHA" ]; then
            cp "$TGT" "$TGT.pre47589.bak"
            cp "$PR47589_SRC" "$TGT"
            rm -f "$VLLM_DIR/compilation/passes/fusion/__pycache__/allreduce_rms_fusion."*.pyc 2>/dev/null || true
            if python3 -c "import py_compile;py_compile.compile('$TGT',doraise=True)" 2>/dev/null; then
                echo "[pr47589] APPLIED overlay -> $TGT (backup .pre47589.bak)"
            else
                echo "[pr47589] ERROR: py_compile failed after overlay; restoring"; cp "$TGT.pre47589.bak" "$TGT"
            fi
        else
            echo "[pr47589] skip: in-image file sha != validated base 93d8f834; re-vendor needed"
        fi
    else
        echo "[pr47589] skip: vllm dir/target not found (VLLM_DIR='${VLLM_DIR:-}')"
    fi
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
