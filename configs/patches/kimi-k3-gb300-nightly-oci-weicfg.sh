#!/usr/bin/env bash
# Kimi-K3 DISAGG on the stock vLLM nightly + Hanjie's patch, WEI'S NUMBERS AS
# WRITTEN (oci-aga). Sibling of kimi-k3-gb300-nightly-disagg.sh.
#
# That script is tuned for lyris and forces three things this one must not,
# because Wei's config is an oci-aga config and the point here is to run it as
# he wrote it:
#
#   segment      lyris tops out at 120 GB/rank (Wei, 2026-08-13: "150gb ... works
#                on OCI-aga, on lyris I can only do 120GB"). His file asks for
#                150, so the value comes from the arm, not from here.
#   MOONCAKE_DEVICE  lyris has mlx5_0..3 and we let auto-discovery read the
#                inventory. His config names mlx5_8, which is an oci-aga NIC --
#                so the arm sets it and this script keeps its hands off. If it
#                is wrong the log says `Found 0 HCAs` and the RDMA topology dump
#                above it names what is actually there.
#   (DURATION stays 3600 -- the campaign standard is mandatory and overrides
#    matching his 1800 window; expect the arm to read low against his chart
#    for that reason alone -- B300 measured -7.8% at 3600 against 1800.)
#
# Everything else -- the patch, the applier, its sha gate, the post-patch
# verification, the mooncake version gate -- is identical.
set -euxo pipefail

export MOONCAKE_VERSION=0.3.12.post1
export FI_VER=${FI_VER:-0.6.16.post3}

# Mooncake host-DRAM segment: defaulted, NOT forced. 600 / 4 ranks = the 150
# GB/rank Wei's config asks for on oci-aga. An arm that needs another value sets
# TOTAL_CPU_DRAM_GB itself.
export TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB:-600}
export MOONCAKE_TP=${MOONCAKE_TP:-4}

# Report the reservation against the tray. The lyris ceiling that this check was
# written for does not apply here -- oci-aga holds 150 GB/rank where lyris
# cannot -- so this warns rather than exits, and MOONCAKE_NODE_FRACTION_PCT
# still tightens it if a tray turns out to be small.
MEM_TOTAL_GB=$(awk '/^MemTotal:/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
if [ -n "${MEM_TOTAL_GB}" ] && [ "${MEM_TOTAL_GB}" -gt 0 ]; then
    MC_CEILING_GB=$(( MEM_TOTAL_GB * ${MOONCAKE_NODE_FRACTION_PCT:-55} / 100 ))
    echo "[mooncake] node MemTotal ${MEM_TOTAL_GB} GB, reserving ${TOTAL_CPU_DRAM_GB} GB" \
         "across ${MOONCAKE_TP} ranks ($(( TOTAL_CPU_DRAM_GB / MOONCAKE_TP )) GB each)," \
         "advisory ceiling ${MC_CEILING_GB} GB"
    if [ "${TOTAL_CPU_DRAM_GB}" -gt "${MC_CEILING_GB}" ]; then
        echo "[mooncake] WARN: ${TOTAL_CPU_DRAM_GB} GB is above ${MC_CEILING_GB} GB." \
             "On lyris that OOM-killed the node; oci-aga is expected to hold it." >&2
    fi
else
    echo "[mooncake] WARN: /proc/meminfo unreadable -- cannot report the reservation"
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
