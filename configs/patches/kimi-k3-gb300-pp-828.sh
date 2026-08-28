#!/usr/bin/env bash
# Kimi-K3 GB300 pipeline-parallel arms (AGG and DISAGG) on the 2026-08-28 nightly.
# =============================================================================
# BASE. vllm/vllm-openai:nightly-6f7df92a8e6cdc74a725b8f10b4d0b48ba2b37ef
# (upstream 6f7df92a, 2026-08-28 04:36 UTC).
#
# THREE PATCHES, IN THIS ORDER, AND ALL THREE ARE REQUIRED.
#
#  1. vllm#54167 -- 'KimiK3LowLatencyLinearMethod' object has no attribute
#     '_gemm_impl'. #50572 made UnquantizedLinearMethod store its GEMM choice in
#     __init__; K3's low-latency mixin never called the inherited initialiser, so
#     any unsupported shape crashes on the fallback. Merged upstream at 08:32
#     UTC, four hours after this nightly was cut, so the image does not have it.
#
#  2. The prefill-checkpoint index cast. _store_cache_checkpoints_kernel
#     (upstream #52789) loads state_idx from an int32 block table and multiplies
#     it by conv_state.stride(0) = 442368; (2**31-1)/442368 = 4854.5, so the
#     product wraps negative at state_idx >= 4855 and the store lands outside
#     any mapping. It surfaces as an illegal memory access about twenty minutes
#     in -- the time allocation takes to climb past block 4854 -- and only on
#     whichever PP stage crosses the threshold first. The 08-19 images we ran
#     before are unaffected only because the kernel does not exist there.
#
#  3. k3-engine-0828.patch -- our own stack, rebased from the 08-19 nightly.
#
# WHY OUR PATCH IS NOT THE 0819 ONE RENAMED. patch(1) puts 31 hunks on the floor
# against this base; a git three-way merge leaves 5 files conflicting, and three
# of those conflicts are real behaviour changes rather than context drift:
#
#   - upstream #53779 introduced kv_cache_config.transfer_groups, a filtered
#     view of kv_cache_groups. _layer_name_to_kv_group_index deliberately still
#     indexes kv_cache_groups: those ids reach _compute_desc_ids as
#     region_group_ids and subscript block_ids[g], which the scheduler hands
#     over in kv_cache_groups coordinates. Filtering there ships another
#     group's blocks, silently.
#   - upstream moved region de-duplication from the per-layer loop to the
#     per-region-spec loop. Keeping our per-layer copy as well would filter
#     twice and drop regions, so only the region_members tracking was carried
#     across into upstream's loop.
#   - block_lens and kv_caches_base_addr are windowed to this worker's PP layer
#     slice; block_strides is newer than our slice (metadata v9) and must be
#     windowed with them or the three lists stop being congruent.
#
# NIXL_CONNECTOR_VERSION is 11: upstream took 8 and 9 for dcp/pcp sizes and
# block_strides, ours moved to 10 and 11.
#
# VERIFIED BEFORE SHIPPING. tests/v1/kv_connector/unit/test_nixl_member_transfer.py
# 20 passed; the five connector suites 244 passed with zero regressions against
# the clean 6f7df92a base (which fails 14 of its own mooncake tests -- our patch
# fixes three of them, all PP).
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
bash /configs/patches/vllm-container-deps-k3-pr54167-828.sh
bash /configs/patches/vllm-container-deps-k3-ckptidx-828.sh

echo "=== k3-pp: apply k3-engine-0819 ==="

PATCH=/configs/patches/k3-engine-0828.patch
[ -f "${PATCH}" ] || PATCH=/configs/k3-engine-0828.patch
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
        echo "[k3-pp] The patch is generated against nightly 5a4c8d9924; if the" \
             "container tag moved, regenerate it from mispa-ms/vllm@misunp/k3-engine-0819." >&2
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

if fail:
    sys.exit("[k3-pp] FATAL:\n  - " + "\n  - ".join(fail))
print("[k3-pp] verified: PP refusal gone, draft PP pinned, aux-over-PP on, "
      "block-table guard in, K3 DCP + partial-hit present, NIXL member identity live")

import vllm
print(f"[k3-pp] vllm {getattr(vllm, '__version__', '?')}")
PY

echo "=== k3-pp: done ==="
