#!/usr/bin/env bash
# The disagg Mooncake setup, the Nsight Systems CLI, and a py-spy stack dumper.
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

# The image already has the NVIDIA cuda repo registered — install directly.
# Do NOT add cuda-keyring on top: it triggers an apt Signed-By conflict.
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    for pkg in nsight-systems-2025.6.3 nsight-systems-2025.3.1 nsight-systems nsight-systems-cli; do
        if apt-get install -y --no-install-recommends "$pkg"; then
            echo "[k3-mooncake-nsys] installed $pkg"
            break
        fi
        echo "[k3-mooncake-nsys] package $pkg unavailable, trying next"
    done
fi

command -v nsys >/dev/null 2>&1 || {
    echo "[k3-mooncake-nsys] FATAL - nsys still not on PATH after install attempts" >&2
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
    nohup setsid bash -c '
      N=0
      while [ "$N" -lt 600 ]; do
        SLOT=$(( N % 30 ))
        rm -f "/logs/pyspy/slot${SLOT}_"*.txt 2>/dev/null || true
        STAMP=$(date -u +%H%M%S)
        for pid in $(pgrep -f "VllmWorker|EngineCore" 2>/dev/null | head -24); do
          timeout 25 py-spy dump --pid "$pid" --nonblocking \
            > "/logs/pyspy/slot${SLOT}_${STAMP}_pid${pid}.txt" 2>&1 || true
          timeout 40 py-spy dump --pid "$pid" --nonblocking --native \
            > "/logs/pyspy/slot${SLOT}_${STAMP}_pid${pid}_native.txt" 2>&1 || true
        done
        N=$(( N + 1 ))
        sleep 30
      done
    ' </dev/null >/logs/pyspy/dumper.log 2>&1 &
    echo "[pyspy] dumper started, writing to /logs/pyspy"
else
    echo "[pyspy] py-spy not on PATH, no stack dumps this run"
fi

echo "=== k3-mooncake-nsys-pyspy: done ==="
