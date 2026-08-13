#!/usr/bin/env bash
# Kimi-K3 DISAGG on the stock vLLM nightly, patched at runtime (GB300).
#
# This replaces kimi-k3-gb300-srcimage-disagg.sh, which needed a 60-minute
# aarch64 image build for every move of wzhao/kimi-k3-agentx-v2 -- and we were
# 139 commits behind it before noticing. Hanjie Qiu reproduced Wei's numbers this
# way on oci-aga, and shared the patch and applier; both are carried verbatim in
# this directory. The vLLM delta is Python only (25 files, no csrc), so nothing
# has to be compiled: the DCP CUDA kernels it depends on are upstream already and
# ship compiled in the nightly.
#
# WHAT THIS DOES NOT CHANGE: the vLLM patch is Hanjie's, byte for byte. Rebasing
# wzhao/kimi-k3-agentx-v2 onto this nightly hits a real semantic conflict --
# upstream grew its own enable_partial_hash_hits and disables it under DCP, while
# Wei's commit is what enables it -- and resolving that ourselves would be
# measuring our merge rather than his stack. His resolution keeps upstream's
# scaffolding, including the unsupported_partial_hit_managers guard, and drops
# only the `dcp_world_size == 1` restriction.
#
# CLUSTER DIFFERENCES, all handled below (this runs on lyris and oci-aga):
#   - NICs are mlx5_0..3, not the mlx5_8 his oci-aga config names. Copying his
#     device_name once already cost us six runs to `Found 0 HCAs`, so this script
#     leaves MOONCAKE_DEVICE unset and lets auto-discovery read the inventory.
#   - the Mooncake segment tops out at 120 GB/rank here, not 150. Wei, 2026-08-13:
#     "150gb global segment size for MC works on OCI-aga, on lyris I can only do
#     120GB". 150 OOM-killed us twice before that was known.
#   - the K3 weights live on a shared path, not in a per-account HF cache.
set -euxo pipefail

export MOONCAKE_VERSION=0.3.12.post1
export FI_VER=${FI_VER:-0.6.16.post3}

# Mooncake host-DRAM segment. vllm-container-deps-mooncake.sh derives the
# per-rank size as TOTAL_CPU_DRAM_GB / MOONCAKE_TP, and MOONCAKE_TP is the ranks
# on this node, so TOTAL_CPU_DRAM_GB is the node-wide reservation: 4 x 120.
#
# Forced, not defaulted. The arms still carry the B300 value in their
# prefill/decode_environment, and a ':-' fallback would silently restore it.
export TOTAL_CPU_DRAM_GB=480
export MOONCAKE_TP=4

# Check it against the tray before the run rather than 40 minutes into it. The
# ceiling is bracketed by measurement: 400 GB of a 955 GB tray runs (62405731),
# 600 OOM-kills (62448615), 748 OOM-killed before that (SLURM 2673272).
MEM_TOTAL_GB=$(awk '/^MemTotal:/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
if [ -n "${MEM_TOTAL_GB}" ] && [ "${MEM_TOTAL_GB}" -gt 0 ]; then
    MC_CEILING_GB=$(( MEM_TOTAL_GB * ${MOONCAKE_NODE_FRACTION_PCT:-55} / 100 ))
    echo "[mooncake] node MemTotal ${MEM_TOTAL_GB} GB, reserving ${TOTAL_CPU_DRAM_GB} GB" \
         "across ${MOONCAKE_TP} ranks ($(( TOTAL_CPU_DRAM_GB / MOONCAKE_TP )) GB each)," \
         "ceiling ${MC_CEILING_GB} GB"
    if [ "${TOTAL_CPU_DRAM_GB}" -gt "${MC_CEILING_GB}" ]; then
        echo "[mooncake] FATAL: ${TOTAL_CPU_DRAM_GB} GB exceeds ${MC_CEILING_GB} GB." >&2
        exit 1
    fi
else
    echo "[mooncake] WARN: /proc/meminfo unreadable -- cannot check the reservation"
fi

# The staged K3 checkpoint sits at a different path on every cluster, and this
# script now runs on more than one of them, so probe rather than hardcode. An
# explicit K3_STAGED_DIR still wins; otherwise take the first candidate that
# exists and say which. Guessing a path costs a whole run to find out, and the
# team sheet has no OCI-aga entry yet.
if [ -z "${K3_STAGED_DIR:-}" ]; then
    for _cand in \
        /lustre/share/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/hanjieq/models/kimi-k3 \
        /lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3
    do
        if [ -d "${_cand}" ]; then
            export K3_STAGED_DIR="${_cand}"
            echo "[k3] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3] WARN: no staged checkpoint found on this cluster; the HF shim will" \
         "fall back to downloading 1.45 TB. Set K3_STAGED_DIR to the local copy." >&2
fi

# hfshim first -- the model has to resolve before anything else matters -- then
# the Mooncake wheel, config and master.
bash /configs/patches/vllm-container-deps-k3-mooncake.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl \
    patch

FI_CUDA=$(python3 -c "
import torch
major = torch.version.cuda.split('.')[0]
print(f'cu{major}' + ('0' if major == '13' else '8'))
")
echo "=== installing flashinfer ${FI_VER} (${FI_CUDA}) ==="
python3 -m pip install --no-deps --force-reinstall "flashinfer-python==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/" "flashinfer-cubin==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/${FI_CUDA}/" \
    "flashinfer-jit-cache==${FI_VER}+${FI_CUDA}" || \
    echo "WARNING: flashinfer-jit-cache ${FI_VER}+${FI_CUDA} not installed; JIT will compile on demand"


# --- KV connectors, the way the image that works installs them -------------
# The stock nightly's NIXL cannot register VRAM on this stack:
#   ucx_utils.cpp:622  4 NVIDIA GPU(s) were detected, but UCX CUDA support was
#                      not found! GPU memory is not supported.
#   nixl_agent.cpp:468 registerMem: registration failed ... all potential backends
#   nixl_cu13._bindings.nixlBackendError: NIXL_ERR_BACKEND
# and UCX confirms it by ignoring UCX_MEMTYPE_REG_WHOLE, which only exists in a
# CUDA-enabled build. Our source-built images never hit this: AIB builds them
# with INSTALL_KV_CONNECTORS=true, and vLLM's Dockerfile then does two steps --
# install requirements/kv_connectors.txt (which pins nixl==1.3.1), then
# force-reinstall nixl-cu<major> over it, with the comment "so the correct
# nixl_ep_cpp.so is installed". Doing only the second is not equivalent.
#
# Mirror .buildkite/scripts/install-kv-connectors.sh, which is what that
# Dockerfile branch actually runs. Two things matter and neither is optional:
#   1. install the generic `nixl` first -- it carries the version the cuXX wheel
#      must match;
#   2. remove BOTH cuda variants before installing the matching one. Upstream:
#      "nixl>=1.1.0 can install multiple CUDA wheel variants. Keep only the
#      variant matching this CI image so nixl_ep_cpp links against the available
#      libcudart." That is the same shape as the mooncake cu12/cu13 shadowing
#      that cost us a day: two distributions, one import, undefined winner.
echo "=== installing KV connectors the way the working image does ==="
python3 -m pip install --no-cache-dir "nixl==${NIXL_VER:-1.3.1}"
KV_META=$(python3 -c "
import importlib.metadata as md
import torch
cuda = torch.version.cuda
if cuda is None:
    raise SystemExit('torch.version.cuda is not set')
print(cuda.split('.', 1)[0], md.version('nixl'))
")
read -r CU_MAJOR NIXL_VERSION <<<"${KV_META}"
python3 -m pip uninstall -y nixl-cu12 nixl-cu13 >/dev/null 2>&1 || true
python3 -m pip install --no-cache-dir --no-deps "nixl-cu${CU_MAJOR}==${NIXL_VERSION}"
python3 -c "
import importlib.metadata as md
import pathlib
import nixl
present = sorted('%s==%s' % (d.metadata['Name'], d.version)
                 for d in md.distributions()
                 if (d.metadata['Name'] or '').startswith('nixl'))
print('  nixl distributions:', present)
print('  imported from:', pathlib.Path(nixl.__file__).parent)
cu = [p for p in present if p.startswith('nixl-cu')]
assert len(cu) == 1, 'expected exactly one nixl-cuXX variant, found %s' % cu
"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

# --- Hanjie's applier, unmodified -----------------------------------------
# It gates on _version.py containing g3d204dfda and refuses to touch any other
# image, which is a stronger identity check than we had: our own source-built
# images could not be told apart until v3 forced the issue. Do not soften it --
# the patch is cut against that exact nightly and a near miss applies with
# offsets rather than failing.
export VLLM_SITE_PACKAGES="$SITE"
export VLLM_DCP_PATCH_FILE=/configs/patches/vllm-wzhao-kimi-k3-agentx-v2-on-nightly-aug13.patch
bash /configs/patches/apply-vllm-kimi-k3-dcp-aug13.sh

# --- wzhao18/vllm@face29e659, on top of Hanjie's patch --------------------
# His branch moved one commit after the patch was cut (2026-08-13 14:38,
# "fix(prefix-cache): preserve DSpark PMU replay state"). Kept as its own file
# rather than folded in, so Hanjie's stays byte-for-byte his and our addition is
# reviewable on its own. Verified: applying the two in order reproduces the
# cherry-picked tree exactly.
#
# Inert on these arms -- it populates eagle_proof_units only for groups with
# use_eagle, and every arm here is no-spec. It matters for DSpark/EAGLE, where
# external lookup must prove the same replay boundary the local prefix cache
# does.
EXTRA_PATCH=/configs/patches/vllm-wzhao-face29e659-dspark-pmu-replay.patch
if [ -r "${EXTRA_PATCH}" ]; then
    if patch --batch --forward --dry-run -d "${SITE}" -p1 < "${EXTRA_PATCH}" >/dev/null 2>&1; then
        patch --batch --forward -d "${SITE}" -p1 < "${EXTRA_PATCH}"
        echo "[patch] applied face29e659 (DSpark PMU replay state)"
    elif patch --batch --reverse --dry-run -d "${SITE}" -p1 < "${EXTRA_PATCH}" >/dev/null 2>&1; then
        echo "[patch] face29e659 already present"
    else
        echo "[patch] FATAL: face29e659 neither applies nor is already applied" >&2
        exit 1
    fi
    python3 -m compileall -q "${SITE}/vllm/distributed/kv_transfer" "${SITE}/vllm/v1/core"
else
    echo "[patch] FATAL: ${EXTRA_PATCH} missing" >&2
    exit 1
fi

echo "=== verifying the patched tree ==="
# Read the tree rather than trust the applier's exit code: `patch --forward`
# reports success when it decides a hunk is already applied, so a partially
# stale tree can exit 0. These assert the behaviours the arms depend on.
python3 -c "
import ast
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
read = lambda rel: (root / rel).read_text()

coord = read('v1/core/kv_cache_coordinator.py')
assert 'def get_replay_boundary' in coord, (
    'the patch did not land: no get_replay_boundary'
)
# The conflict Hanjie resolved: upstream disables fine-grained hits under DCP,
# Wei's change is what enables them. Every arm here is DCP8, so a silent revert
# to the upstream form would turn prefix-match-unit off and look like a plain
# regression.
assert 'enable_partial_hash_hits = has_partial_mamba_group' in coord, (
    'partial hash hits are still gated on dcp_world_size == 1: the DCP arms '
    'would run with prefix-match-unit silently inert'
)
assert 'dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size' in coord, (
    'the DCP branch of has_partial_mamba_group is missing'
)

proof = read('distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py')
assert 'eagle_proof_units' in proof, (
    'wzhao18/vllm@face29e659 did not land: external EAGLE lookup will not prove '
    'the replay boundary the local prefix cache enforces'
)

stkcm = read('v1/core/single_type_kv_cache_manager.py')
assert '(block_idx, req_blocks[block_idx])' in stkcm, (
    'the partial-hit CoW still records new_computed_blocks[-1]; '
    'allocate_new_blocks dies on a bare AssertionError under load'
)

# vllm#51733, the MLA prefill workspace fix Wei named as the CUDA-OOM guard.
# It is upstream and therefore in the nightly, but it is the one thing whose
# absence explains every weicfg arm we lost, so check rather than assume.
mla = read('model_executor/layers/attention/mla_attention.py')
assert 'max_num_seqs' not in mla.split('def align_mla_chunked_context_workspace_size')[1].split('def ')[0], (
    'MLA prefill workspace still scales with max_num_seqs: vllm#51733 is absent '
    'and the prefill worker will OOM at gmu 0.92'
)
print('=== patched tree verified ===')
"

echo "=== mooncake client/master version agreement ==="
python3 - <<'MCGATE'
import importlib.metadata as md
import os
import sys

want = os.environ.get("MOONCAKE_VERSION", "0.3.11.post1")
try:
    import mooncake
    from mooncake.store import MooncakeDistributedStore  # noqa: F401
except Exception as exc:
    sys.exit("mooncake is not importable after setup: %s" % exc)

dists = [d for d in md.distributions()
         if (d.metadata["Name"] or "").startswith("mooncake-transfer-engine")]
present = sorted("%s==%s" % (d.metadata["Name"], d.version) for d in dists)
print("  imported from: %s" % getattr(mooncake, "__file__", "?"))
print("  distributions present: %s" % (present or "none"))
if len(dists) != 1:
    sys.exit("expected exactly one mooncake-transfer-engine distribution, found "
             "%d: %s" % (len(dists), present))
if dists[0].version != want:
    sys.exit("mooncake is %s, not the pinned %s -- the deps script fell back"
             % (present[0], want))
print("  ok: %s" % present[0])
MCGATE

echo "=== GB300 nightly disagg setup complete ==="
