#!/usr/bin/env bash
# Kimi-K3: LMCache node-local KV pool (KV OFFLOAD = LMCache).
# =============================================================================
# WHY: LMCacheMPConnector needs a companion `lmcache server` process that owns
#   the shared CPU-DRAM KV pool, started BEFORE `vllm serve`. srt-slurm has no
#   notion of a per-node companion process for vLLM, and the worker preamble
#   chains setup_script && ... && vllm serve in one srun step, so the server is
#   started here in the background and this script blocks until it is reachable.
#   Returning early would let vLLM connect to a pool that is not up yet.
#
# One pool per node (not per-rank segments), which is the whole point of the
# LMCache variant versus SimpleCPUOffloadConnector.
#
# Chains the k3-hfshim first: the HF cache still has to point at the pre-staged
# 1.5 TB checkpoint regardless of which offload backend is in use.
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps-k3-hfshim.sh ]]; then
    bash /configs/patches/vllm-container-deps-k3-hfshim.sh
else
    echo "[k3-lmcache] FATAL: k3-hfshim not found; the checkpoint would not resolve." >&2
    exit 1
fi

LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.1}"
# Must match lmcache.mp.port in the vLLM --kv-transfer-config.
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
# Recipe guidance: size the pool to ~75% of the host DRAM you can dedicate. bia
# nodes are 2014 GiB and our verified ceiling for a pinned pool there is ~1500 GiB
# (2500 GiB OOM'd), so 75% of 1500 = 1125. The pool grows lazily from l1-init-size-gb.
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-1125}"
LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
LMCACHE_READY_TIMEOUT="${LMCACHE_READY_TIMEOUT:-600}"

echo "=== k3-lmcache: starting lmcache server (l1=${LMCACHE_L1_SIZE_GB} GB, port ${LMCACHE_PORT}) ==="

uv pip install --system "lmcache==${LMCACHE_VERSION}" 2>/dev/null \
  || pip install "lmcache==${LMCACHE_VERSION}"

LOG=/logs/lmcache-server.log
mkdir -p /logs

# --max-gpu-workers 1 avoids concurrent-GPU-transfer stalls under heavy
# async-load pressure; CPU-side workers stay at 8.
nohup lmcache server \
    --port "${LMCACHE_PORT}" \
    --http-port "${LMCACHE_HTTP_PORT}" \
    --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
    --l1-init-size-gb "${LMCACHE_L1_INIT_SIZE_GB}" \
    --max-gpu-workers 1 \
    --max-cpu-workers 8 \
    --chunk-size 1024 \
    --l1-align-bytes 16384 \
    --eviction-trigger-watermark 0.85 \
    --eviction-ratio 0.10 \
    --eviction-policy LRU \
    --supported-transfer-mode lmcache_driven \
    > "${LOG}" 2>&1 &
LMCACHE_PID=$!
echo "[k3-lmcache] lmcache server pid=${LMCACHE_PID}, log=${LOG}"

# Block until the pool answers, otherwise vLLM races it and fails to attach.
deadline=$(( SECONDS + LMCACHE_READY_TIMEOUT ))
until curl -sf "http://127.0.0.1:${LMCACHE_HTTP_PORT}/" >/dev/null 2>&1; do
    if ! kill -0 "${LMCACHE_PID}" 2>/dev/null; then
        echo "[k3-lmcache] FATAL: lmcache server exited during startup. Tail of ${LOG}:" >&2
        tail -40 "${LOG}" >&2 || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "[k3-lmcache] FATAL: lmcache server not ready after ${LMCACHE_READY_TIMEOUT}s. Tail of ${LOG}:" >&2
        tail -40 "${LOG}" >&2 || true
        exit 1
    fi
    sleep 5
done

echo "=== k3-lmcache: server ready on ${LMCACHE_PORT} (http ${LMCACHE_HTTP_PORT}) ==="
