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

# ── Can this container's toolchain even target this GPU ──────
# The single most expensive thing we got wrong on Hecate. Pipeline 64809381
# loaded 192.85 GiB/rank of weights over 176 s and only then died on
#     ptxas fatal : Value 'sm_107a' is not defined for option 'gpu-name'
# because the image shipped CUDA 13.0.88, whose ptxas has no sm_107 target at
# all (13.3.73 has none either; 13.4 is the first that does). Every Triton JIT
# in the process dies the same way, so nothing downstream can work.
#
# One ptxas invocation on a four-line PTX file answers it in milliseconds.
# Fail here rather than forty minutes into weight loading.
echo "=== toolchain: can ptxas target this device? ==="
CAP=$(python3 -c "import torch;print('%d%d' % torch.cuda.get_device_capability(0))" 2>/dev/null || echo "")
if [ -z "$CAP" ]; then
    echo "WARNING: could not read device capability from torch; skipping the ptxas check"
else
    PTXAS="${TRITON_PTXAS_PATH:-$(command -v ptxas || echo /usr/local/cuda/bin/ptxas)}"
    echo "  device capability sm_${CAP}, testing $PTXAS"
    "$PTXAS" --version 2>&1 | tail -1
    printf '.version 8.0\n.target sm_90\n.address_size 64\n.visible .entry _nop() { ret; }\n' > /tmp/_cap_probe.ptx
    if ! "$PTXAS" "--gpu-name=sm_${CAP}a" /tmp/_cap_probe.ptx -o /tmp/_cap_probe.o 2>/tmp/_cap_probe.err; then
        echo "FATAL: $PTXAS cannot target sm_${CAP}a:"
        sed 's/^/       /' /tmp/_cap_probe.err
        echo "       This container's CUDA predates the GPU. Every Triton JIT will"
        echo "       fail the same way, after the model has already been loaded."
        echo "       Use an image whose CUDA knows sm_${CAP} -- on Hecate that is"
        echo "       gitlab-master.nvidia.com:5005/dl/dgx/vllm:rubin-py3-devel, a"
        echo "       nightly (CUDA 13.5 as of 2026-08-26). Or install a 13.4+"
        echo "       ptxas and point TRITON_PTXAS_PATH at it, which fixes Triton"
        echo "       but not the CuTe DSL paths."
        exit 1
    fi
    echo "  OK: $PTXAS targets sm_${CAP}a"
fi

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
#
# The budget is a FRACTION of the node, not a fixed reserve. An earlier form
# demanded MemTotal >= KV + 200, which is not scale-free: on a 93 GB node it
# rejects a 1 GB request and then advises a ceiling of "-107 GB". 85% is what
# the B300 baseline this arm is derived from actually runs (1700 GB on a ~2 TB
# node), so it is the precedent as well as the sane shape.
KV_PER_NODE_GB="${K3_KV_OFFLOAD_PER_NODE_GB:-${K3_KV_OFFLOAD_GB:-600}}"
MEM_TOTAL_GB=$(awk '/^MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
KV_CEILING_GB=$(( MEM_TOTAL_GB * 85 / 100 ))
echo "host MemTotal ${MEM_TOTAL_GB} GB; this node's share of the KV tier is ${KV_PER_NODE_GB} GB (ceiling ${KV_CEILING_GB} GB = 85%)"
if [ "$KV_PER_NODE_GB" -gt "$KV_CEILING_GB" ]; then
    echo "FATAL: this node's share of the host KV tier is ${KV_PER_NODE_GB} GB but the"
    echo "       node has only ${MEM_TOTAL_GB} GB, so 85% is ${KV_CEILING_GB} GB. Lower"
    echo "       kv-offloading-size and TOTAL_CPU_DRAM_GB (both server-wide) so that"
    echo "       the per-node share is at most ${KV_CEILING_GB} GB."
    exit 1
fi

# ── 3. Wire the HF cache to JET's shared copy ────────────────
# srt-slurm has no model_paths alias here, so hf:moonshotai/Kimi-K3 resolves
# through HF_HOME. Left to itself vLLM pulls the repo from the Hub -- which on
# Hecate takes about 13 minutes, not the hours the runbook warns about, so the
# cost is not walltime. The cost is CORRECTNESS: the Hub's main moves, and a
# fresh pull gets whatever it is today (a590ce09 on 2026-08-26) while every
# other cluster in this comparison runs 9f62e4e. bia's own cache says so:
#   /lustre/fsw/coreai_comparch_infbench/common/cache/hub/
#       models--moonshotai--Kimi-K3/refs/main = 9f62e4e9fffbd0a83ddd60e1c209d828994b3569
#
# So point at the JET artifact instead of downloading. JET syncs this repo to
# every cluster we use, hecate_lustre included, and it carries 9f62e4e. The
# shim materialises HF cache layout over the staged directory; K3_PIN_SHA makes
# it label the entry with the revision the weights actually are rather than the
# Hub's current main.
#
# The shim fails loudly if K3_STAGED_DIR is absent, and proves the wiring with
# an offline snapshot_download before returning. It also runs
# vllm-container-deps.sh, which installs numactl and msgpack -- the only two
# packages this arm adds. No framework version is touched; in particular there
# is no FlashInfer pin, which is the point of running stock nightly.
echo "=== Kimi-K3 checkpoint: wiring HF cache to the staged copy ==="
HF_HOME="${HF_HOME:?HF_HOME must be set by the config}"
: "${K3_STAGED_DIR:?K3_STAGED_DIR must be set by the config (the JET artifact path)}"
bash /configs/patches/vllm-container-deps-k3-hfshim.sh

FOUND="${HF_HOME}/hub/models--moonshotai--Kimi-K3"
# du the STAGED directory, not the cache entry. The entry is a symlink farm, so
# plain du reports ~0 and `du -L` double-counts JET's own internal symlinks --
# it read 2908G against a 1.5T tree on 64802405. The staged dir is the real
# number and the only one worth printing.
SZ=$(du -sBG "$K3_STAGED_DIR" 2>/dev/null | awk '{print $1}')
echo "  cache entry: $FOUND"
echo "  refs/main:   $(cat "$FOUND/refs/main" 2>/dev/null || echo '<none>')"
echo "  staged dir:  $K3_STAGED_DIR  ($SZ)"

# A directory can exist and hold nothing but a failed partial download.
#
# DERIVE THE EXPECTED COUNT, DO NOT HARDCODE IT. An earlier revision of this
# script asserted `shards >= 100` on the belief that "a complete K3 checkpoint
# has several hundred". It has 96. That guess failed three healthy jobs on a
# fully staged 1454 GB checkpoint. Shard filenames are self-describing --
# model-00001-of-00096.safetensors -- so the file tree states its own expected
# count and no number has to be remembered here.
# find -L, NOT find. The shim wires snapshots/<sha> as a SYMLINK to the staged
# directory, and plain find does not descend into symlinks -- it would report
# zero shards on a perfectly good checkpoint and fail the job for it.
SHARDS=$(find -L "$FOUND" -name '*.safetensors' 2>/dev/null | wc -l)
INCOMPLETE=$(find -L "$FOUND" -name '*.incomplete' 2>/dev/null | wc -l)
# -print -quit, NOT `| head -1`. Under this script's `set -o pipefail`, head
# closing the pipe after one line sends find SIGPIPE, the pipeline reports 141,
# and -e kills the worker -- which is exactly how pipeline 64802405 died
# ("Critical process 'agg_0_...' exited with code 141") on a checkpoint that was
# wired correctly. find stops itself with -quit, so nothing gets signalled.
EXPECTED=$(find -L "$FOUND" -name '*-of-*.safetensors' -print -quit 2>/dev/null \
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
