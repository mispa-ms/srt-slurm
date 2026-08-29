#!/usr/bin/env bash
# Kimi-K3 GB300 pipeline-parallel arms (AGG and DISAGG) on the 2026-08-29 nightly.
# =============================================================================
# BASE. vllm/vllm-openai:nightly-6d4562c59b97b4e35d459ff9389e71b6fe4995de
# (upstream 6d4562c5, 2026-08-29 06:09 UTC).
#
# WHAT CHANGED FROM THE 08-28 CHAIN. Two of the four patches are gone because
# the nightly now carries them, and one of ours was handed back to upstream:
#
#  - vllm#54167 (K3 low-latency GEMM fallback init) MERGED as f956e1c34 and is
#    an ancestor of this nightly. The 08-28 chain needed it because the fix
#    landed four hours after that image was cut. Script deleted.
#
#  - vllm#53324 (MooncakeStore under hybrid DCP) MERGED as 7fd9cc036, also an
#    ancestor. Our own connector.py and worker.py changes -- 139 lines that put
#    every group's block size into the scheduler's coordinate space and lifted
#    the hybrid+DCP refusal -- are dropped in favour of it. Upstream's rule is
#    resolve_dcp_kv_block_size: scale ATTENTION groups by dcp, leave every other
#    group alone. That is exactly our multi-group branch. The one divergence is
#    the single-group case, which we special-cased and upstream does not, and
#    which K3 never reaches: a hybrid model always has more than one group.
#    Our mooncake/store/coordinator.py work STAYS -- it is the exact-boundary
#    retry for partial hits, a different defect that #53324 does not touch.
#
#  - Mamba align retention is now upstream's #53803 instead of ours. See
#    vllm-container-deps-k3-pr53803-829.sh for why, and note that our
#    single_type_kv_cache_manager.py delta drops from 411 lines to 21: all that
#    survives is the _apply_cow log-and-continue guard, which is not retention
#    at all but a crash guard. Its assert killed three otherwise-healthy arms
#    and fired zero times on the same config once it stopped being fatal.
#
# FOUR STEPS, IN THIS ORDER.
#
#  1. vllm#53803 -- mamba align retention checkpoints (open upstream, carries
#     #53798). Shipped as a resolved merge, not the raw PR diff.
#
#  2. The prefill-checkpoint index cast. _store_cache_checkpoints_kernel in
#     models/kimi_k3/nvidia/kda.py loads state_idx from an int32 block table and
#     multiplies it by state_stride_0 = 442368; (2**31-1)/442368 = 4854.5, so the
#     product wraps negative at state_idx >= 4855 and the store lands outside any
#     mapping. It surfaces as an illegal memory access about twenty minutes in --
#     the time allocation takes to climb past block 4854 -- and only on whichever
#     PP stage crosses the threshold first. STILL REQUIRED: the cast is not in
#     this nightly.
#
#  3. The revert of vllm#52388 (upstream PR #53774, still open). #52388's fused
#     multi-group Mamba kernel caches raw block-table pointers taken during the
#     temporary CUDA-graph memory-profile capture; the later real capture then
#     dereferences addresses freed with that allocation. Here it lands first as
#     'expected 3 block tables, got 4' out of compile_or_warm_up_model.
#     STILL REQUIRED: #53774 has not merged.
#
#  4. k3-engine-0829.patch -- our own stack. 23 files, down from 25 and from
#     3,906 lines to 2,207. The NIXL push member-identity work applied to this
#     nightly with zero conflicts.
#
# WHY PUSH AND NOT PULL. Upstream #53360 lifts the pull-path PP guards, which
# would be the other way to get PP on a Mamba hybrid. It is open, +1843/-167,
# and its own description says it does not duplicate the push work. Measured at
# PP1 the two connectors are indistinguishable (push 1,510 vs pull 1,512 at c64,
# 1,428 vs 1,450 at c48) -- but PP1 leaves the PP machinery inactive, so that
# comparison cannot discriminate. Push is what is verified at PP2 here.
#
# APPLY ORDER IS LOAD-BEARING. kda_metadata.py is touched by both step 3 and
# step 4; the engine patch is cut against the base plus step 1 only, so step 4
# must come last. Verified by replaying all four in this order onto a pristine
# 6d4562c5 tree: zero rejects, two hunks absorbed at offsets -2 and -12.
# =============================================================================
set -euo pipefail

# The staged checkpoint sits at a different path on every cluster. An explicit
# K3_STAGED_DIR wins; otherwise take the first candidate that exists and say
# which one, because guessing costs a whole run to find out.
if [ -z "${K3_STAGED_DIR:-}" ]; then
    for _cand in \
        /lustre/share/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/hanjieq/models/kimi-k3 \
        /lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3
    do
        if [ -d "${_cand}" ]; then
            export K3_STAGED_DIR="${_cand}"
            echo "[k3-pp] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3-pp] FATAL: no staged checkpoint on this cluster. Set K3_STAGED_DIR." >&2
    echo "[k3-pp] Refusing to continue -- the fallback is a 1.45 TB download inside" \
         "a 4-hour job, which fails later and less legibly than this does." >&2
    exit 1
fi

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-k3-pr53803-829.sh
bash /configs/patches/vllm-container-deps-k3-ckptidx-829.sh
bash /configs/patches/vllm-container-deps-k3-revert52388-829.sh

echo "=== k3-pp: apply k3-engine-0829 ==="

PATCH=/configs/patches/k3-engine-0829.patch
[ -f "${PATCH}" ] || PATCH=/configs/k3-engine-0829.patch
if [ ! -f "${PATCH}" ]; then
    echo "[k3-pp] FATAL: patch not found at /configs/patches/ or /configs/" >&2
    exit 1
fi

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[k3-pp] vllm root: ${VLLM_ROOT}"

if grep -q "The drafter is instantiated only on the last pipeline stage" \
        "${VLLM_ROOT}/vllm/config/speculative.py"; then
    echo "[k3-pp] already applied"
else
    if ! command -v patch >/dev/null 2>&1; then
        echo "[k3-pp] installing patch(1)"
        apt-get update -qq && apt-get install -y -qq patch
    fi
    # Dry-run first. A partially applied engine is worse than a failed job: it
    # starts, serves, and is wrong in a way no perf metric shows.
    if ! patch -p1 -d "${VLLM_ROOT}" --batch --dry-run --forward < "${PATCH}" > /tmp/k3-all-dry.log 2>&1; then
        echo "[k3-pp] FATAL: patch does not apply to this image." >&2
        echo "[k3-pp] The patch is generated against nightly 6d4562c5 + #53803; if the" \
             "container tag moved, regenerate it from srt-slurm@misunp/k3-gb300-0829." >&2
        cat /tmp/k3-all-dry.log >&2
        exit 1
    fi
    # --batch: without it, a patch naming a file the image does not have
    # prompts "File to patch:" and reads the answer from stdin -- which is
    # the patch itself. It then eats patch lines as answers and applies a
    # truncated diff while reporting success.
    patch -p1 -d "${VLLM_ROOT}" --batch --forward < "${PATCH}"
    echo "[k3-pp] applied"
fi

# Assert by value, not by presence. A verifier that asks hasattr of an attribute
# set in __init__ is always False on the class and killed three arms at startup
# before anyone noticed it could not pass.
python3 - <<'PY'
import importlib.util, os, sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
def src(p):
    return open(os.path.join(root, p)).read()

fail = []

if "DSpark does not support pipeline parallelism" in src(
        "vllm/v1/worker/gpu/spec_decode/dspark/utils.py"):
    fail.append("the upstream DSpark-under-PP refusal is still present")
if "pipeline_parallel_size=1" not in src("vllm/config/speculative.py"):
    fail.append("draft ParallelConfig still inherits the target PP size")
if "supports_aux_hidden_states_over_pp" not in src("vllm/models/kimi_k3/nvidia/model.py"):
    fail.append("K3 does not opt in to relaying aux hidden states across PP stages")
if "_check_block_table_width" not in src("vllm/models/kimi_k3/nvidia/kda_metadata.py"):
    fail.append("KDA block-table width guard missing")

# These two are upstream's, not ours, and this arm depends on both. #50484 gave
# K3 DCP; #50493 made partial prefix-cache hits work under it, without which
# prefix matching silently drops to the DCP-scaled group block.
# The NIXL member-identity path. _tracks_region_members is the gate that
# register_kv_caches uses to advertise per-region layer members; without it a
# K3 PP producer hands its peer an empty region_members and the handshake
# fails. It existed once as a second, drifted copy of the routing gate.
nixl_worker = src("vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py")
if "_tracks_region_members" not in nixl_worker:
    fail.append("NIXL member-identity registration gate missing")
if "does not support pipeline_parallel_size > 1 with Mamba" in nixl_worker:
    fail.append("the Mamba-under-PP refusal is still present in the NIXL worker")
if not os.path.exists(os.path.join(
        root, "vllm/distributed/kv_transfer/kv_connector/v1/nixl/member_transfer.py")):
    fail.append("member_transfer.py (the transfer planner) is missing")

# Decode-side PP. Counting completions per overlapping producer stage is only
# half of it -- the transfer must also reach every decode stage, or a
# PP-sharded consumer waits forever on notifs nobody sends. Assert both halves
# and the absence of the refusal that stood in for them, because an arm that
# comes up and hangs at the first request is the expensive way to find out.
push_worker = src("vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py")
if "_overlapping_remote_pp_ranks" not in nixl_worker:
    fail.append("the overlapping-stage rule is missing from the NIXL worker")
if "decode_pp_size" not in src(
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py"):
    fail.append("PUSH_REG does not advertise decode_pp_size; P cannot see D's split")
if "decode_pp_size" not in push_worker:
    fail.append("the push write path does not read decode_pp_size")
if "not supported yet (got" in push_worker:
    fail.append("the decode-side PP refusal is still present in the push worker")

if "MLADCPManager" not in src("vllm/models/kimi_k3/nvidia/mla.py"):
    fail.append("K3 DCP (#50484) missing from this image")
if "dcp_world_size > 1" not in src("vllm/v1/core/kv_cache_coordinator.py"):
    fail.append("DCP-aware partial hash hits (#50493) missing from this image")

# The three pieces this chain hands to upstream, each verified by a symbol
# rather than by whether its patch step said "applied". patch(1) reports
# "Reversed (or previously applied)" whenever context has merely drifted, so
# a step can pass while having written nothing.
stkcm = src("vllm/v1/core/single_type_kv_cache_manager.py")
if "retention_grid_block" not in stkcm:
    fail.append("#53803 retention is not in this tree; sparse retention will "
                "register grid blocks the align allocator has already nulled")
if "_cow_slot_mismatches" not in stkcm:
    fail.append("the _apply_cow log-and-continue guard is missing; its assert "
                "killed three arms on DCP8 partial hits")
if "resolve_dcp_kv_block_size" not in src("vllm/v1/core/kv_cache_utils.py"):
    fail.append("#53324 mooncake-under-DCP is not in this image; our own "
                "connector/worker changes were dropped in favour of it")
# Dropping our mooncake worker.py in favour of #53324 was right; dropping the
# whole connector.py with it was not, and cost pipeline 65185599 at engine
# init. The base PP-aware setter rejects pp_rank > 0 to protect connectors that
# read peer handshake metadata; this one meets its peers in the store and
# throws the value away, so without the override every PP2 arm dies with
# "MooncakeStoreConnector received pp_rank > 0 handshake metadata".
if "set_xfer_handshake_metadata_pp_aware" not in src(
        "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py"):
    fail.append("the MooncakeStore PP handshake override is missing; every "
                "prefill-PP arm will die at engine core init")
if "to(tl.int64)" not in src("vllm/models/kimi_k3/nvidia/kda.py"):
    fail.append("the checkpoint-index int64 cast is missing; the store wraps "
                "negative past state_idx 4854 and faults ~20 min in")

if fail:
    sys.exit("[k3-pp] FATAL:\n  - " + "\n  - ".join(fail))
print("[k3-pp] verified: PP refusal gone, draft PP pinned, aux-over-PP on, "
      "block-table guard in, K3 DCP + partial-hit present, NIXL member identity live")

import vllm
print(f"[k3-pp] vllm {getattr(vllm, '__version__', '?')}")
PY

echo "=== k3-pp: done ==="
