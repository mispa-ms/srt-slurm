#!/usr/bin/env bash
# The disagg Mooncake setup, nsys, py-spy, the diagnostics, and the actual fix.
# =============================================================================
# kimi-k3-v2-ds-prefix-cache-mooncake.sh with one addition: nsys.
#
# vllm/vllm-openai ships no nsys, and srtctl prefixes the worker launch with
# `nsys profile ...` whenever profiling.type is set. Without the CLI the launch
# exits 127 with "nsys: command not found" and the job dies during bring-up, so
# a profiling config on the plain Mooncake script cannot work at all.
#
# The install block is the same newest-first probe as
# vllm-container-deps-k3-nsys.sh, which is the AGG-side script this mirrors. It
# fails loudly here rather than 127 lines later inside the worker.
#
# Everything above it is unchanged from the Mooncake script: the CUDA 13
# transfer engine, SA's V2 DS prefix-cache backport (vLLM #49291 / #50153) and
# their hybrid KV-load recovery.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl

python3 -m pip install msgpack
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 || true
python3 -m pip install --no-deps 'mooncake-transfer-engine-cuda13==0.3.12.post1'

python3 /configs/patches/patch_kimi_k3_v2_ds_prefix_cache.py
python3 /configs/patches/patch_kimi_k3_mooncake_hma_recompute.py

# Log ring-buffer indices next to the shm_broadcast long-wait warning, and make
# the indefinite readers warn too. py-spy showed both sides parked in
# acquire_read; current_idx is per-process local state it cannot reach.
python3 /configs/patches/patch_shm_broadcast_index_debug.py

# Count commands sent against responses written on both sides. Arm J showed the
# queues are healthy and the engine is simply owed a reply that was never made;
# these four counters say which side lost it. Must run after the index patch --
# it extends the warning that patch introduces.
python3 /configs/patches/patch_multiproc_rpc_accounting.py

# The fix. The worker AsyncOutputCopy thread parks in copy_event.synchronize()
# after an nsys capture and never returns, so responses stop reaching the engine
# while every counter still looks healthy. Poll cudaEventQuery instead of waiting
# on cudaEventBlockingSync, and fail loudly on a deadline.
python3 /configs/patches/patch_async_output_event_poll.py

# The image already has the NVIDIA cuda repo registered — install directly.
# Do NOT add cuda-keyring on top: it triggers an apt Signed-By conflict.
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    for pkg in nsight-systems-2025.6.3 nsight-systems-2025.3.1 nsight-systems nsight-systems-cli; do
        if apt-get install -y --no-install-recommends "$pkg"; then
            echo "[k3-mooncake-evtfix] installed $pkg"
            break
        fi
        echo "[k3-mooncake-evtfix] package $pkg unavailable, trying next"
    done
fi

command -v nsys >/dev/null 2>&1 || {
    echo "[k3-mooncake-evtfix] FATAL - nsys still not on PATH after install attempts" >&2
    exit 1
}
nsys --version
# ---------------------------------------------------------------------------
# py-spy stack dumper.
#
# After an nsys capture the decode worker stops answering the executor and the
# engine dies 1800 s later on "RPC call to sample_tokens timed out". Six runs in,
# we can say what it is not -- buffer volume, GPU metrics, KV transfer errors,
# stall length, the RPC timeout, DSpark -- but not what it is, because nothing
# in vLLM dumps the stack of a worker that stops responding.
#
# So attach from outside. This runs in the same container as the worker (the
# setup script's own output lands in the worker log), and py-spy reads another
# process's stack without stopping it, which matters -- we are diagnosing a hang,
# not creating one.
#
# Ring of 30 slots at 30 s keeps the last ~15 minutes. The wedge lasts 1800 s
# before the engine gives up, so whatever is in the final slots is the hang
# itself, not the approach to it.
# ---------------------------------------------------------------------------
python3 -m pip install --no-deps py-spy || echo "[pyspy] install failed, continuing"

if command -v py-spy >/dev/null 2>&1; then
    mkdir -p /logs/pyspy
    # Match on the process title vLLM actually sets -- set_process_title writes
    # "VLLM::<name>" (system_utils.py), so the workers are VLLM::Worker_TP0 and
    # so on. The first attempt matched "VllmWorker", which is the log prefix and
    # not a process name, and caught nothing but this loop itself: pgrep -f sees
    # our own command line, so the pattern matched a fresh subshell every cycle
    # and every real worker was missed. PYSPY_SELF is how we exclude ourselves.
    nohup setsid bash -c '
      N=0
      while [ "$N" -lt 600 ]; do
        SLOT=$(( N % 30 ))
        rm -f "/logs/pyspy/slot${SLOT}_"*.txt 2>/dev/null || true
        STAMP=$(date -u +%H%M%S)
        K=0
        for pid in $(pgrep -f "VLLM::" 2>/dev/null); do
          if grep -qa PYSPY_SELF "/proc/${pid}/cmdline" 2>/dev/null; then continue; fi
          TITLE=$(tr -d "\0" < "/proc/${pid}/cmdline" 2>/dev/null | cut -c1-40)
          timeout 25 py-spy dump --pid "$pid" --nonblocking \
            > "/logs/pyspy/slot${SLOT}_${STAMP}_pid${pid}.txt" 2>&1 || true
          echo "$TITLE" >> "/logs/pyspy/slot${SLOT}_${STAMP}_pid${pid}.txt"
          if [ "$K" -lt 3 ]; then
            timeout 60 py-spy dump --pid "$pid" --nonblocking --native \
              > "/logs/pyspy/slot${SLOT}_${STAMP}_pid${pid}_native.txt" 2>&1 || true
          fi
          K=$(( K + 1 ))
        done
        echo "[pyspy] cycle $N slot $SLOT dumped $K process(es)" >> /logs/pyspy/dumper.log
        N=$(( N + 1 ))
        sleep 30
      done
    ' </dev/null >>/logs/pyspy/dumper.log 2>&1 &
    echo "[pyspy] dumper started (PYSPY_SELF), writing to /logs/pyspy"
else
    echo "[pyspy] py-spy not on PATH, no stack dumps this run"
fi

echo "=== k3-mooncake-nsys-shmdbg: done ==="
