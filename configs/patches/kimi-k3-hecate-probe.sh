#!/bin/bash
# Kimi-K3 bring-up on Hecate / VR200, on the stock vLLM main nightly.
#
# This is NOT a patch script. It installs nothing and edits no source. Its whole
# job is to answer, in the first two minutes of the job, the questions that
# CLUSTER_RUNBOOK.md says are cluster-shaped and must be re-derived per cluster --
# and to FAIL LOUDLY on the one that would otherwise burn the whole allocation.
#
# WHY IT EXISTS. Every K3 recipe we have runs on B300 (x86, 8 GPU/node, bia) or
# GB300 (aarch64, 4 GPU/node, lyris/oci-aga/aws-cmh). Hecate is the first VR200
# cluster and nothing about it has been measured by us. The runbook's rule is:
#
#   "Staged checkpoints live at a different path per cluster. Probe a candidate
#    list and fail loudly if none match -- the fallback is a 1.45 TB download
#    inside a 4-hour job."
#
# The checkpoint probe below is that. The rest is free diagnostics: whatever
# this prints is what run 2 gets to stop guessing about.
#
# WHAT IS DELIBERATELY NOT HERE, and why:
#
#   FlashInfer pin.  Every B300 arm runs kimi-k3-nightly-fi0616rc5.sh, which
#   force-reinstalls flashinfer-python/cubin/jit-cache at 0.6.16rc5 for the
#   MegaMoE kernels. flashinfer-jit-cache is a CUDA- AND ARCH-specific wheel;
#   there is no evidence one exists for aarch64 + Rubin, and a failed pin leaves
#   a half-downgraded FlashInfer that is worse than the one the container
#   shipped. So this arm runs whatever the nightly ships and PRINTS it, which is
#   the fact we actually need before deciding.
#
#   Mooncake.  Needs device_name, and RDMA device names are the single most
#   cluster-shaped field there is (runbook s2). We do not know Hecate's, so the
#   probe reports ibv_devices and this arm uses vLLM's native host tier instead.
#
#   Any vLLM patch.  #51295, the MLA workspace patch and the hybrid-KV recompute
#   patch all exist to make speculation and Mooncake survivable. Neither is on
#   this arm, so neither patch is either. Stock main is the point.

set -euo pipefail

echo "############################################################"
echo "# Kimi-K3 / Hecate VR200 bring-up probe"
echo "############################################################"

# ── 1. What are we actually running on ───────────────────────
echo "=== host / arch ==="
uname -a
echo "python machine: $(python3 -c 'import platform; print(platform.machine())')"

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv || true
echo "GPU count: $(nvidia-smi -L 2>/dev/null | wc -l)"

echo "=== NVLink / NIC topology ==="
# Which HCA names exist here decides what a future Mooncake arm may put in
# device_name. Empty output means the container cannot see the fabric at all,
# which is itself the answer.
nvidia-smi topo -m 2>/dev/null || echo "WARNING: nvidia-smi topo unavailable"
ibv_devices 2>/dev/null || echo "WARNING: ibv_devices unavailable (no RDMA visible in container)"

echo "=== host DRAM ==="
grep -E '^(MemTotal|MemAvailable)' /proc/meminfo

# ── 2. Host DRAM must cover the offload tier we asked for ────
# The AgentX number is a coverage curve, y = P/(1-h)/N_gpu, so the host KV tier
# is not a second-order knob -- it moves h, which moves the headline. A request
# the node cannot back is not rejected at start; it is OOM-killed mid-run. Check
# it here, while failing is still cheap.
#
# THE UNIT MATTERS AND IS NOT OBVIOUS. vLLM's kv-offloading-size is documented
# in config/cache.py:241 as "the total buffer size summed across ALL TP ranks"
# -- server-wide, not per node. This script runs on each worker node, so the
# number to check against /proc/meminfo is the server-wide total divided by the
# node count, which the config passes as K3_KV_OFFLOAD_PER_NODE_GB. Comparing
# the server-wide figure against one node's MemTotal would reject configurations
# that fit, which is how the first Hecate submission would have been mis-sized
# in the other direction.
KV_PER_NODE_GB="${K3_KV_OFFLOAD_PER_NODE_GB:-${K3_KV_OFFLOAD_GB:-600}}"
MEM_TOTAL_GB=$(awk '/^MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "host MemTotal ${MEM_TOTAL_GB} GB; this node's share of the KV tier is ${KV_PER_NODE_GB} GB"
if [ "$MEM_TOTAL_GB" -lt "$((KV_PER_NODE_GB + 200))" ]; then
    echo "FATAL: node has ${MEM_TOTAL_GB} GB; ${KV_PER_NODE_GB} GB of host KV plus"
    echo "       ~200 GB of headroom does not fit. Lower kv-offloading-size and"
    echo "       TOTAL_CPU_DRAM_GB (both server-wide) so that the per-node share"
    echo "       is at most $((MEM_TOTAL_GB - 200)) GB."
    exit 1
fi

# ── 3. The checkpoint, or nothing ────────────────────────────
# srt-slurm has no model_paths alias on this cluster, so hf:moonshotai/Kimi-K3
# resolves through HF_HOME. If it is not staged there, vLLM starts a 1.4 TB
# download inside a 5-hour job and the job dies with a timeout that names
# nothing. Fail here instead, naming every path tried.
echo "=== Kimi-K3 checkpoint probe ==="
HF_HOME="${HF_HOME:?HF_HOME must be set by the config}"
CANDIDATES=(
    "${HF_HOME}/hub/models--moonshotai--Kimi-K3"
    "${HF_HOME}/models--moonshotai--Kimi-K3"
    "/lustre/fsw/${SLURM_PPP:-}/common/cache/hub/models--moonshotai--Kimi-K3"
    "/scratch/models/Kimi-K3"
    "/models/Kimi-K3"
)
FOUND=""
for c in "${CANDIDATES[@]}"; do
    if [ -d "$c" ]; then
        SZ=$(du -sBG "$c" 2>/dev/null | awk '{print $1}')
        echo "  present: $c  ($SZ)"
        FOUND="$c"
        break
    fi
    echo "  absent : $c"
done

if [ -z "$FOUND" ]; then
    echo
    echo "FATAL: Kimi-K3 is not staged on this cluster."
    echo "  HF_HOME=${HF_HOME}"
    echo "  Without it vLLM downloads ~1.4 TB inside the job and the allocation"
    echo "  expires before the server is ever healthy. Stage the checkpoint (JET"
    echo "  distribute, or an out-of-band hf download into HF_HOME) and resubmit."
    echo "  Paths tried are listed above."
    exit 1
fi

# A directory can exist and hold nothing but a failed partial download.
#
# DERIVE THE EXPECTED COUNT, DO NOT HARDCODE IT. An earlier revision of this
# script asserted `shards >= 100` on the belief that "a complete K3 checkpoint
# has several hundred". It has 96. That guess failed three healthy jobs on a
# fully staged 1454 GB checkpoint. Shard filenames are self-describing --
# model-00001-of-00096.safetensors -- so the file tree states its own expected
# count and no number has to be remembered here.
SHARDS=$(find "$FOUND" -name '*.safetensors' 2>/dev/null | wc -l)
INCOMPLETE=$(find "$FOUND" -name '*.incomplete' 2>/dev/null | wc -l)
EXPECTED=$(find "$FOUND" -name '*-of-*.safetensors' 2>/dev/null | head -1 \
           | sed -nE 's/.*-of-0*([0-9]+)\.safetensors$/\1/p')
echo "  safetensors shards: ${SHARDS} (filenames declare ${EXPECTED:-unknown}), .incomplete files: ${INCOMPLETE}"

if [ "$INCOMPLETE" -gt 0 ]; then
    echo "FATAL: ${INCOMPLETE} .incomplete file(s) under ${FOUND} -- a download is"
    echo "       in progress or was interrupted. Not a staged model."
    exit 1
fi
if [ -n "$EXPECTED" ]; then
    if [ "$SHARDS" -lt "$EXPECTED" ]; then
        echo "FATAL: ${SHARDS} of ${EXPECTED} safetensors shards present under ${FOUND}."
        echo "       The filenames declare ${EXPECTED}; this checkpoint is truncated."
        exit 1
    fi
elif [ "$SHARDS" -lt 2 ]; then
    # No -of- pattern to read, so fall back to the weakest defensible claim:
    # a 2.8T model is never one shard.
    echo "FATAL: ${SHARDS} safetensors file(s) under ${FOUND} and no -of-N naming"
    echo "       to check against. This is not a sharded Kimi-K3 checkpoint."
    exit 1
fi

# ── 4. Does this container's vLLM actually know K3 ───────────
# Kimi-K3 landed on vLLM main in PR #50000 (2026-07-30) as
# KimiK3ForConditionalGeneration -> vllm.models.kimi_k3. A nightly older than
# that, or an aarch64 build that dropped the out-of-tree package, fails at model
# load 40 minutes in. One import answers it now.
echo "=== vLLM / FlashInfer as shipped by this container ==="
python3 -m pip list 2>/dev/null | grep -iE "flashinfer|^vllm |^torch " || true
python3 - <<'PY'
import sys
from vllm.model_executor.models.registry import ModelRegistry
arches = set(ModelRegistry.get_supported_archs())
want = "KimiK3ForConditionalGeneration"
print("vLLM registry knows", want, ":", want in arches)
if want not in arches:
    sys.exit(f"FATAL: {want} is not registered in this container's vLLM. "
             "Kimi-K3 needs a nightly at or after PR #50000 (2026-07-30).")
import vllm
print("vllm", vllm.__version__)
try:
    import flashinfer
    print("flashinfer", flashinfer.__version__)
except Exception as e:                                  # noqa: BLE001
    print("flashinfer import failed:", e)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device capability", torch.cuda.get_device_capability(0))
PY

echo "############################################################"
echo "# probe passed -- checkpoint staged, K3 registered, DRAM fits"
echo "############################################################"
