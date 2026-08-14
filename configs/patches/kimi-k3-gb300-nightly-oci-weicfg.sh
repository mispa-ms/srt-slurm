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


# --- NIXL, shared with the other GB300 nightly setup script ----------------
bash /configs/patches/install-nixl-cu13.sh

# --- UCX / CUDA, shared with the other GB300 nightly setup script ---------
# Both scripts hit the same oci-aga failure, and the last four runs were lost
# to fixing one copy of a duplicated block. There is one copy now.
bash /configs/patches/ucx-cuda-check.sh

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

# Print the root cause LAST. srt-slurm quotes only the final 50 lines of a
# failed process log, and a vLLM engine failure ends with forty lines of
# asyncio/uvloop frames under "Engine core initialization failed. See root
# cause above" -- so the exception that explains it is pushed out of view
# every time. sitecustomize is imported automatically by CPython, so this
# covers every worker without touching a launch command.
cp /configs/patches/k3_roothook.py "${SITE}/k3_roothook.py"
# A .pth, not sitecustomize.py: site.py imports only the first
# sitecustomize on sys.path and the image already ships one at
# /usr/lib/python3.12/sitecustomize.py, which would have shadowed ours
# without a word. A .pth import line runs for every path entry.
echo "import k3_roothook" > "${SITE}/zz-k3-roothook.pth"
# A WARNING, never fatal. The first version of this check ran `import
# sitecustomize` and asserted, which failed on a healthy install and took
# four weiport arms down with it -- the same shape as the UCX guard that
# killed eight arms this morning. A diagnostic that cannot confirm itself
# must not end the run.
if python3 -c "import sys; raise SystemExit(0 if sys.excepthook.__name__ == 'hook' else 1)" 2>/dev/null; then
    echo "[roothook] installed"
else
    echo "[roothook] WARNING: not confirmed active; tracebacks may stay truncated" >&2
fi

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
# Site 1: how much of a hit is reported. A .patch, cut against this tree.
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
# Site 2: which keys are asked for. A string replacement rather than a diff --
# two hand-made patches for this one hunk failed for reasons unrelated to the
# change (wrong base tree, then a malformed @@ header), each costing a submit.
python3 /configs/patches/apply-lookup-align.py "${SITE}" || exit 1
# And one sample key from each side. The master says 19,224 keys are resident
# and Get never fires, so written and queried keys disagree; a key is a string,
# so print one of each and read the difference instead of deriving it.
python3 /configs/patches/apply-keysample-log.py "${SITE}" || exit 1
# vllm#50359, as we carried it on misunp/k3-merged-mooncake-dcp and then left
# behind when this track moved to the nightly. It revalidates that a reported
# partial-hash hit names an object the store actually holds, and shortens by
# one alignment unit until it does -- "a longer stored key proves one object
# exists at that endpoint; it does not imply independently addressable objects
# at every shorter hash boundary". That is the shape of our 0.0%: the lookup
# asks about 240,000 keys per window, reports no error, and never issues a
# Get. Measured on B300 c8 DCP=8 before: 2,757,664 OBJECT_NOT_FOUND.
#
# Cut against the real target tree (nightly 3d204dfda + Hanjie + face29e659 +
# our alignment work), not against repo/vllm -- applying the branch commit
# directly lands with fuzz 2 and a 44-line offset.
REVAL_PATCH=/configs/patches/vllm-mooncake-50359-exact-hit-revalidate.patch
if patch --batch --forward --dry-run -d "${SITE}" -p1 < "${REVAL_PATCH}" >/dev/null 2>&1; then
    patch --batch --forward -d "${SITE}" -p1 < "${REVAL_PATCH}"
    echo "[patch] applied: #50359 exact-hit revalidation"
elif patch --batch --reverse --dry-run -d "${SITE}" -p1 < "${REVAL_PATCH}" >/dev/null 2>&1; then
    echo "[patch] #50359 revalidation already present"
else
    echo "[patch] FATAL: the #50359 revalidation patch does not apply" >&2
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
