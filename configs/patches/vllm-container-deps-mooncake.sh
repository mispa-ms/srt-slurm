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

# ---- Transport + kernel tuning (best-effort; we have container priv on bia) -----
# Protocol is env-driven: rdma (Wei's InfX B300 path, scales — no per-transfer
# sockets) vs tcp (no RDMA MR-registration, but exhausts ephemeral ports at high
# transfer volume). Default rdma.
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
# RDMA device pin (root cause of bia EFAULT per Cyrus Chang / InferenceX): empty
# device_name lets mooncake auto-discover, which on a multi-NIC host grabs the wrong
# (limited-connection) RNIC -> ibv_reg_mr EFAULT. Pin the primary RNIC explicitly.
# bia RNICs are mlx5_* (see srt-slurm CLAUDE.md SGLang path); "1st NIC" = mlx5_0.
# Comma-list allowed (e.g. "mlx5_0,mlx5_1"). Empty = legacy auto-discover.
MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-}"
# Dump the node's RDMA topology so we can SEE which RNICs exist + GPU affinity
# (ground truth for picking device_name; reusable on GB200/GB300).
echo "[mooncake] ---- RDMA topology (for device_name selection) ----"
ibv_devices 2>/dev/null || echo "[mooncake]   (ibv_devices unavailable)"
ibstat -l 2>/dev/null | sed 's/^/[mooncake]   ibstat: /' || true
nvidia-smi topo -m 2>/dev/null | sed 's/^/[mooncake]   topo: /' || true
echo "[mooncake] MOONCAKE_DEVICE=${MOONCAKE_DEVICE:-<auto/empty>} protocol=${MOONCAKE_PROTOCOL}"
echo "[mooncake] ----------------------------------------------------"
# RDMA registration fixes (mooncake troubleshooting docs): ibv_reg_mr of the large
# host offload buffer EFAULTed on bia. vm.max_map_count default (65530) is too small
# for many MRs; memlock cap blocks pinning. Both best-effort.
sysctl -w vm.max_map_count=16777216 2>/dev/null && echo "[mooncake] raised vm.max_map_count" || echo "[mooncake] (could not set vm.max_map_count)"
ulimit -l unlimited 2>/dev/null && echo "[mooncake] memlock unlimited" || echo "[mooncake] (could not raise memlock — relies on cluster default)"
# TCP ephemeral-port fixes (only relevant when MOONCAKE_PROTOCOL=tcp).
sysctl -w net.ipv4.ip_local_port_range="1024 65535" 2>/dev/null && echo "[mooncake] widened ip_local_port_range" || true
sysctl -w net.ipv4.tcp_tw_reuse=1 2>/dev/null && echo "[mooncake] enabled tcp_tw_reuse" || true

# ---- Mooncake store install (cu13 wheel for B300/GB300) ----------------------
MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.11.post1}"
# Pick the wheel by (arch, CUDA major), then let pip decide whether it INSTALLS.
#
# The wheel has to satisfy two independent things: glibc (does it install) and
# libcudart.so.<major> (does it import). CUDA major is a property of the image,
# so it is tested here. glibc is NOT -- it is a property of the wheel, and it
# moves per release:
#
#   mooncake-transfer-engine-cuda13   0.3.11.post1  manylinux_2_39_aarch64
#                                     0.3.12.post1  manylinux_2_28_aarch64
#
# This block used to hardcode "glibc < 2.39 -> fall back". That was correct at
# the 0.3.11.post1 pin and became wrong the moment we pinned 0.3.12.post1: on
# GB300 (glibc 2.35) it rejected a wheel that installs fine, and silently ran
# mooncake-transfer-engine 0.3.9 instead. 0.3.9's register_buffer is not what
# vLLM's MooncakeStoreConnector calls -- 96 registrations per worker rejected
# with ErrorCode::INVALID_PARAMS (-600), then 82,760 AddressNotRegistered, and
# an offload tier that moved zero bytes while every arm was still labelled
# "dram-mooncake".
#
# So do not encode a floor at all. Ask pip -- it knows the tags of the wheel it
# is actually being asked for. Ask it by attempting the install rather than with
# --dry-run: --dry-run needs pip>=22.2, and a probe that errors for the wrong
# reason would fall back silently, which is the exact failure being fixed here.
# A platform mismatch is resolver-stage, so it costs a metadata fetch, not a
# 100 MB wheel.
CU_MAJOR=$(ldconfig -p 2>/dev/null | grep -oE 'libcudart\.so\.[0-9]+' | grep -oE '[0-9]+$' | sort -un | tail -1)
GLIBC_MINOR=$(getconf GNU_LIBC_VERSION 2>/dev/null | grep -oE '[0-9]+$')
echo "[mooncake] arch=$(uname -m) cuda_major=${CU_MAJOR:-unknown} glibc=2.${GLIBC_MINOR:-?}"

# Optionally keep what the image already ships.
#
# Hanjie's InferenceMAX submission (NVIDIA/InferenceMAX#213) runs the same
# nightly image and does NOT reinstall mooncake -- "0.3.12.post1 (CUDA13,
# image-resident, no reinstall)" -- and his offload reads: 73.9% external hit at
# disagg c48, 2,565 GB put against 491 GB get. Ours reads 0.0% on the same image
# and the same patch set, having replaced the resident wheel with the PyPI one
# of the same version. Same version is not the same build, and RDMA memory
# registration is exactly the kind of thing that differs between them.
#
# Off by default: the reinstall is what fixed the silent 0.3.9 downgrade, and
# every number we have was measured with it.
if [ "${K3_MOONCAKE_USE_IMAGE:-0}" = "1" ]; then
    if python3 -c "import mooncake" >/dev/null 2>&1; then
        _IMGV=$(pip show mooncake-transfer-engine-cuda13 2>/dev/null | awk '/^Version:/{print $2}')
        [ -z "${_IMGV}" ] && _IMGV=$(pip show mooncake-transfer-engine 2>/dev/null | awk '/^Version:/{print $2}')
        echo "[mooncake] K3_MOONCAKE_USE_IMAGE=1 -- keeping the image's mooncake ${_IMGV:-<unknown>}"
        MOONCAKE_INSTALLED=1
        SKIP_MOONCAKE_INSTALL=1
        # Everything the skipped block would have defined. `set -u` is on, and
        # the shim branch below reads these unconditionally.
        NEED_CUDART12_SHIM=0
        MOONCAKE_PKG="mooncake-transfer-engine-cuda13==${_IMGV:-image}"
        MOONCAKE_FALLBACK_PKG="${MOONCAKE_PKG}"
    else
        echo "[mooncake] FATAL: K3_MOONCAKE_USE_IMAGE=1 but the image has no mooncake" >&2
        exit 1
    fi
fi

# Clear BOTH distributions first. They provide the same `mooncake` package, so
# with both installed --force-reinstall only overwrites files and leaves the
# loser's metadata behind: `pip show <name>` then answers for a version that is
# not the one importing. Uninstalling afterwards is worse -- it deletes the files
# the winner just wrote. Start from zero and there is one answer.
if [ "${SKIP_MOONCAKE_INSTALL:-0}" != "1" ]; then
for _mc in mooncake-transfer-engine-cuda13 mooncake-transfer-engine; do
    if pip show "${_mc}" >/dev/null 2>&1; then
        echo "[mooncake] removing pre-existing ${_mc}"
        pip uninstall --quiet -y "${_mc}" >/dev/null 2>&1 || true
    fi
done

MOONCAKE_FALLBACK_PKG="mooncake-transfer-engine==${MOONCAKE_AARCH_VERSION:-0.3.9}"
NEED_CUDART12_SHIM=0
MOONCAKE_INSTALLED=0
if [ "$(uname -m)" = "aarch64" ] && [ "${CU_MAJOR}" = "12" ]; then
    # aarch64 + cu12 image: the cuda13 wheel would install and then fail to
    # import (libcudart.so.13). Not a glibc question -- go straight to non-cu13.
    MOONCAKE_PKG="${MOONCAKE_FALLBACK_PKG}"
else
    MOONCAKE_PKG="mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
    echo "[mooncake] trying ${MOONCAKE_PKG}"
    if pip install --quiet --no-cache-dir --no-deps --force-reinstall "${MOONCAKE_PKG}"; then
        MOONCAKE_INSTALLED=1
    else
        echo "[mooncake] pip rejected ${MOONCAKE_PKG} on this platform (glibc 2.${GLIBC_MINOR:-?})"
        echo "[mooncake] falling back to ${MOONCAKE_FALLBACK_PKG} + cu12 runtime shim"
        MOONCAKE_PKG="${MOONCAKE_FALLBACK_PKG}"
        [ "${CU_MAJOR}" = "13" ] && NEED_CUDART12_SHIM=1
    fi
fi
fi   # SKIP_MOONCAKE_INSTALL
if [ "${NEED_CUDART12_SHIM}" = "1" ]; then
    echo "[mooncake] cu13 image + non-cu13 wheel -> installing cu12 runtime shim (libcudart.so.12)"
    pip install --quiet --no-cache-dir "nvidia-cuda-runtime-cu12"
    # Locate libcudart.so.12 via pip's install Location (importing the `nvidia` namespace
    # package is unreliable — it has no importable __init__), then a filesystem fallback.
    CUDART12_SO=""
    LOC=$(pip show nvidia-cuda-runtime-cu12 2>/dev/null | awk -F': ' '/^Location:/{print $2}')
    if [ -n "${LOC}" ]; then
        CUDART12_SO=$(ls "${LOC}"/nvidia/cuda_runtime/lib/libcudart.so.12* 2>/dev/null | head -1)
    fi
    if [ -z "${CUDART12_SO}" ]; then
        CUDART12_SO=$(find / -maxdepth 9 -path '*cuda_runtime*' -name 'libcudart.so.12*' 2>/dev/null | head -1)
    fi
    if [ -n "${CUDART12_SO}" ] && [ -f "${CUDART12_SO}" ]; then
        # cp into a default loader-path dir + ldconfig so mooncake's .so finds it at
        # dlopen time in the WORKER process (LD_LIBRARY_PATH set here would not survive
        # into the separate worker srun; the ld.so cache does).
        cp -f "${CUDART12_SO}" /usr/local/lib/ 2>/dev/null || cp -f "${CUDART12_SO}" /usr/lib/
        ldconfig 2>/dev/null || true
        echo "[mooncake] installed libcudart.so.12 shim: $(basename "${CUDART12_SO}") -> ld cache"
    else
        echo "[mooncake] WARN: could not locate cu12 libcudart.so.12 — mooncake import may fail"
    fi
fi
if [ "${MOONCAKE_INSTALLED}" != "1" ]; then
    echo "[mooncake] installing ${MOONCAKE_PKG}"
    pip install --quiet --no-cache-dir --no-deps --force-reinstall "${MOONCAKE_PKG}"
fi
python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
pip list 2>/dev/null | grep -i "^mooncake" | sed 's/^/[mooncake] present: /'
echo "[mooncake] installed ${MOONCAKE_PKG}"

# Bound MooncakeStoreConnector transfer batches (InferenceX patch). Mooncake's TCP
# connection pool grows without a ceiling, so big agentic per-layer transfers
# exhaust the node's TCP ports ("connect: Cannot assign requested address"). The
# patch splits batch_put/get into INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS-sized
# chunks. Idempotent; auto-finds vllm's mooncake store worker.py.
if [ -f /configs/patches/patch_vllm_mooncake_transfer_batches.py ]; then
    python3 /configs/patches/patch_vllm_mooncake_transfer_batches.py \
        && echo "[mooncake] transfer-batching patch applied" \
        || echo "[mooncake] WARN: transfer-batching patch failed (worker.py anchors may differ from this vLLM)"
fi

# ---- Mooncake config -----------------------------------------------------------
# bia B300 host RAM = 2014 GiB; use a 1500 GB aggregate CPU pool (~500 GiB headroom).
# Mooncake embedded mode: each of TP ranks contributes one global segment to the
# shared local store, so pre-divide by TP. TP8 -> 1500/8 = 187 GB/rank.
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1500}"
TP="${MOONCAKE_TP:-8}"
PER_RANK_GB=$(( TOTAL_CPU_DRAM_GB / TP ))
MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-/tmp/mooncake_config.json}"
# Into /logs when the run has one: only that directory is harvested with the job, and
# the master's periodic store stats (fill/keys/evictions) are the only record of what
# the host-DRAM pool actually did. Earlier runs wrote to /tmp and lost them.
if [[ -z "${MOONCAKE_MASTER_LOG:-}" && -d /logs ]]; then
    MOONCAKE_MASTER_LOG=/logs/mooncake_master.log
fi
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

# srt-slurm's native `mooncake_kv_store` block launches the master on the infra node
# and writes its own store config into the log dir. In that mode this script must only
# install the wheel and leave both alone, or the two masters race for ports and the
# worker reads the wrong config. Set MOONCAKE_SKIP_MASTER=1 to stop here.
if [ "${MOONCAKE_SKIP_MASTER:-0}" = "1" ]; then
    echo "[mooncake] MOONCAKE_SKIP_MASTER=1: wheel installed; master and config"
    echo "[mooncake] left to srt-slurm's mooncake_kv_store block"
    exit 0
fi

cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:${MOONCAKE_MASTER_PORT}",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "4GB",
  "protocol": "${MOONCAKE_PROTOCOL}",
  "device_name": "${MOONCAKE_DEVICE}",
  "enable_offload": false
}
EOF
echo "[mooncake] wrote $MOONCAKE_CONFIG_PATH (per-rank ${PER_RANK_GB}GB x TP${TP})"

# ---- Launch local mooncake_master (detached so it survives this script) -------
echo "[mooncake] starting master rpc=127.0.0.1:${MOONCAKE_MASTER_PORT} metrics=${MOONCAKE_METRICS_PORT}"
# --default_kv_lease_ttl is a uint64 (SECONDS) in mooncake 0.3.9 (the aarch64/cu13 wheel);
# 0.3.11 (bia x86) accepts a "1h" duration string. Normalize to seconds so both work — and
# because the backend env (MOONCAKE_KV_LEASE_TTL) does NOT reach this master-launch shell,
# so a "1h" default here would fail 0.3.9 with "illegal value '1h' for uint64 flag".
_ttl_raw="${MOONCAKE_KV_LEASE_TTL:-3600}"
case "${_ttl_raw}" in
    *h) MOONCAKE_TTL_SECS=$(( ${_ttl_raw%h} * 3600 ));;
    *m) MOONCAKE_TTL_SECS=$(( ${_ttl_raw%m} * 60 ));;
    *s) MOONCAKE_TTL_SECS="${_ttl_raw%s}";;
    *)  MOONCAKE_TTL_SECS="${_ttl_raw}";;
esac
echo "[mooncake] default_kv_lease_ttl=${MOONCAKE_TTL_SECS}s (from '${_ttl_raw}')"
# The admin/metrics HTTP server is mandatory (master.cpp:1100-1105 → exit on bind
# failure) and cannot be disabled by any flag; enable_metric_reporting only gates
# the logging thread. So both ports must be genuinely free — see the probe above.
setsid nohup mooncake_master --port "${MOONCAKE_MASTER_PORT}" \
    --metrics_port="${MOONCAKE_METRICS_PORT}" \
    --default_kv_lease_ttl="${MOONCAKE_TTL_SECS}" \
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
