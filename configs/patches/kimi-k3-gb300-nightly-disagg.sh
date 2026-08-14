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
# 1.3.2, not the 1.3.1 that requirements/kv_connectors.txt pins.
#
# The cu13 wheel's bundled UCX is version-dependent, and 1.3.1's is byte
# identical to nixl-cu12's -- a CUDA 12 UCX inside the CUDA 13 wheel. auditwheel
# names the libraries by content hash, so this is checkable:
#
#   nixl-cu13==1.3.0  libucp-fb7bfdea.so.0.0.0
#   nixl-cu13==1.3.1  libucp-fb7bfdea.so.0.0.0   <- same as cu12
#   nixl-cu13==1.3.2  libucp-e76cb9e6.so.0.0.0
#   nixl-cu12==1.3.1  libucp-fb7bfdea.so.0.0.0
#
# That is why UCX loads no CUDA component here while every library resolves and
# UCX_MODULE_DIR points at the right place: the component is not built for this
# CUDA. Hanjie's container carries 1.3.2, which is the difference between his
# oci-aga runs working and ours not -- his squashfs has
# nixl_cu13.libs/libucp-e76cb9e6 while ours has fb7bfdea.
#
# Following .buildkite/scripts/install-kv-connectors.sh to the letter is what
# pinned us to the broken pair, since it takes the version from the generic
# `nixl` requirement.
python3 -m pip install --no-cache-dir "nixl==${NIXL_VER:-1.3.2}"
KV_META=$(python3 -c "
import importlib.metadata as md
import torch
cuda = torch.version.cuda
if cuda is None:
    raise SystemExit('torch.version.cuda is not set')
print(cuda.split('.', 1)[0], md.version('nixl'))
")
read -r CU_MAJOR NIXL_VERSION <<<"${KV_META}"
# Do NOT uninstall the other variant first. install-kv-connectors.sh does
# ("Keep only the variant matching this CI image"), and following it is what
# broke us: Hanjie's working container has BOTH nixl_cu12.libs and
# nixl_cu13.libs present, which is what vLLM's Dockerfile actually produces --
# `nixl` pulls cu12, then cu13 is force-reinstalled over it, and nothing is
# removed. His runs show 40 "Backend UCX was instantiated" and zero "UCX CUDA
# support was not found"; ours, with the same nixl 1.3.2 and cu12 removed,
# report the warning on every rank.
#
# So the variant that must be present is not only the matching one. Leave both.
python3 -m pip install --no-cache-dir --force-reinstall --no-deps "nixl-cu${CU_MAJOR}==${NIXL_VERSION}"
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
assert any(p.startswith('nixl-cu13') for p in cu), (
    'nixl-cu13 is missing; UCX will have no CUDA-13 component: %s' % present)
"


# --- why UCX may not see CUDA -------------------------------------------
# nixl's wheel does ship the CUDA plugins (libuct_cuda.so, libucm_cuda.so), so
# "UCX CUDA support was not found" is a load failure, not a missing build. Their
# NEEDED list is the tell:
#   libuct_cuda.so -> libcuda.so.1, libnvidia-ml.so.1
#   libucm_cuda.so -> libcuda.so.1, libcudart.so.13
# torch only needs libcuda + libcudart, which is why the model loads and KV cache
# is sized normally while UCX still refuses VRAM. libnvidia-ml.so.1 is injected by
# the container runtime only when NVIDIA_DRIVER_CAPABILITIES includes `utility`.
# Print the loader's view so the next failure names the missing library instead
# of leaving us to infer it.
echo "=== driver libraries the UCX CUDA plugin needs ==="
echo "  NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
for _lib in libcuda.so.1 libnvidia-ml.so.1 libcudart.so.13; do
    _found=$(ldconfig -p 2>/dev/null | grep -m1 "${_lib}" || true)
    if [ -n "${_found}" ]; then
        echo "  ok      ${_lib}: ${_found##*=> }"
    else
        echo "  MISSING ${_lib}"
    fi
done
_plugdir=$(python3 -c "
import pathlib
try:
    import nixl
except Exception:
    raise SystemExit(0)
for p in pathlib.Path(nixl.__file__).parent.parent.glob('nixl*.libs/ucx'):
    print(p); break
" 2>/dev/null)
if [ -n "${_plugdir}" ]; then
    echo "  ucx plugin dir: ${_plugdir}"
    for _p in "${_plugdir}"/libuct_cuda.so "${_plugdir}"/libucm_cuda.so; do
        [ -e "${_p}" ] && echo "    present: $(basename ${_p})"
    done
    # ldd resolves exactly what the loader will, including the injected driver.
    ldd "${_plugdir}/libuct_cuda.so" 2>/dev/null | grep -E "not found|libcuda|libnvidia-ml" | sed 's/^/    /' || true
    # UCX searches its compiled-in default for loadable modules -- libucs
    # carries "MODULE_DIR ... /usr/lib64/ucx ... Directory to search for
    # loadable modules" -- but auditwheel relocated the plugins under the
    # wheel's own .libs. The directory it looks in does not exist here, so
    # libuct_cuda.so is present and never opened, which is what "UCX CUDA
    # support was not found" means while ldd resolves every dependency.
    #
    # Wei does not hit this: his NGC container has a properly installed UCX at
    # /usr/local/ucx, where the compiled-in prefix is right.
    #
    # The fix is UCX_MODULE_DIR, and it has to be set in the ARM's environment,
    # not exported here: workers are launched by their own srun and do not
    # inherit this shell. Exporting it would look like a fix and change nothing
    # -- the same trap as the mooncake cudart shim, which had to go through the
    # ld.so cache for exactly this reason. So verify instead, and fail loudly
    # when the arm's value does not match what is on disk.
    if [ -z "${UCX_MODULE_DIR:-}" ]; then
        echo "[ucx] FATAL: UCX_MODULE_DIR is unset in the worker environment." >&2
        echo "[ucx]   set it in the arm to: ${_plugdir}" >&2
        exit 1
    fi
    if [ "${UCX_MODULE_DIR}" != "${_plugdir}" ]; then
        echo "[ucx] FATAL: UCX_MODULE_DIR=${UCX_MODULE_DIR} but the plugins are" \
             "at ${_plugdir}. UCX would search the wrong directory and report" \
             "'CUDA support was not found' with every library present." >&2
        exit 1
    fi
    echo "  UCX_MODULE_DIR=${UCX_MODULE_DIR} (matches the plugin dir)"
    # The cu13 wheel has shipped a cu12 UCX before (1.3.0 and 1.3.1 both bundle
    # nixl-cu12's libucp). Catch that here rather than 40 minutes later as
    # "UCX CUDA support was not found" with every library resolving.
    _ucp=$(ls "${_plugdir}"/../libucp-*.so.* 2>/dev/null | head -1)
    if [ -n "${_ucp}" ]; then
        echo "    bundled UCX: $(basename "${_ucp}")"
        case "$(basename "${_ucp}")" in
            libucp-fb7bfdea.*)
                echo "[ucx] FATAL: this is nixl-cu12's UCX inside the cu13 wheel" \
                     "(nixl-cu13 1.3.0/1.3.1). Its CUDA component will not load on a" \
                     "CUDA 13 image. Pin NIXL_VER=1.3.2 or later." >&2
                exit 1 ;;
        esac
    fi
    # Is there a system UCX in this image at all? The wheel's copy is the only
    # one vLLM's Dockerfile installs, but a base image or a transitive dep (e.g.
    # OpenMPI) can drag one in -- and a properly installed UCX has its CUDA
    # component where its own prefix expects it. If one exists, pointing at it
    # is a better answer than routing NIXL around UCX, which changes the
    # transport and therefore the numbers.
    echo "  system UCX (outside the wheel):"
    ldconfig -p 2>/dev/null | grep -E "libucp|libuct|libucs" | grep -v "nixl" | sed 's/^/    /' || echo "    none"
    for _d in /usr/lib64/ucx /usr/lib/aarch64-linux-gnu/ucx /usr/local/ucx/lib/ucx /opt/hpcx/ucx/lib/ucx; do
        [ -d "${_d}" ] && echo "    module dir exists: ${_d} ($(ls "${_d}" 2>/dev/null | grep -c cuda) cuda modules)"
    done
fi

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

# --- ours: block-align the external hit ------------------------------------
# Not from Wei or Hanjie. The mooncake store keys one object per block_size
# chunk, so the only lengths it can serve are multiples of the block size, but
# the lookup aligns to hash_block_size whenever partial hash hits are on. The
# reported hit then lands where nothing was written, the load finds nothing, and
# nothing errors -- which is why every K3 DCP run here writes gigabytes and
# reads back 0.0% external hit while save_put_failed_keys stays 0.
#
# configs/patches/test_mooncake_dcp_keyset.py has been failing on exactly this
# at dcp=2/4/8 (and passing at dcp=1). I dismissed it twice as a stale test
# before connecting it to the 0%.
#
# Reverting this restores the sub-block tail from vllm#49502; measure before
# keeping either way.
ALIGN_PATCH=/configs/patches/vllm-mooncake-external-hit-block-aligned.patch
if patch --batch --forward --dry-run -d "${SITE}" -p1 < "${ALIGN_PATCH}" >/dev/null 2>&1; then
    patch --batch --forward -d "${SITE}" -p1 < "${ALIGN_PATCH}"
    echo "[patch] applied: external hit is block-aligned"
elif patch --batch --reverse --dry-run -d "${SITE}" -p1 < "${ALIGN_PATCH}" >/dev/null 2>&1; then
    echo "[patch] external-hit alignment already present"
else
    echo "[patch] FATAL: the external-hit alignment patch does not apply" >&2
    exit 1
fi
python3 -m compileall -q "${SITE}/vllm/distributed/kv_transfer"

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
