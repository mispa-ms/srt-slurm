#!/usr/bin/env bash
# The 2026-08-31 nightly (44fe2a392b) carrying upstream PRs only, plus three lines.
# =============================================================================
# WHY THIS CHAIN EXISTS. Every other chain on this branch carries patches we wrote.
# This one carries almost nothing of ours: it exists to answer "what is actually
# still missing from upstream", by running the three K3 targets on a base that is
# effectively current main and seeing what breaks.
#
# THE BASE. nightly-44fe2a392b71d52a8d72faf2f8278834379482c9, cut 2026-08-31 06:17Z.
# That is 33 commits behind main HEAD e9dd6d4834 (2026-08-31 09:17 PDT) and none of
# the 33 touch K3, MLA, DCP, mamba, or any KV connector. So "runs on this image" and
# "runs on current main" are the same claim, and no source build is needed.
#
# WHAT UPSTREAM ABSORBED SINCE THE 08/28 CHAINS. Verified file by file against the
# 44fe2a392b tree, not assumed:
#
#   pr54167       skips   -- super().__init__() is in low_latency_gemm.py:728
#   53324         skips   -- connector.py:112 refuses only `pcp > 1`; the DCP half of
#                            the refusal is gone, and worker.py resolves the per-group
#                            spec through resolve_dcp_kv_cache_spec(spec, dcp_size).
#                            Our three-file Mooncake carry has nothing left to add.
#   54044         skips   -- cudagraph_utils.py:867 resets _mamba_group_ids/_mamba_spec
#                            on profiling teardown. The cache is still unkeyed, but the
#                            window our `mambacache` carry closed is closed upstream.
#   dspark-pp     REPLACED by #50514 -- see below. Do not apply our 596-line carry.
#   interleave    REPLACED by #54457 -- see below. Do not apply our name-list carry.
#
# WHAT WE APPLY, AND WHY IT IS NOT OURS:
#
#   1. upstream-pr54457.patch   #54457 by cjackal, open. Adds
#      `requires_dcp_block_aligned_interleave` to KVConnectorBase_V1 so #50611 stops
#      pinning cp_kv_cache_interleave_size for connectors that move no KV between
#      instances. This is the predicate our own `interleave` carry said upstream was
#      missing, implemented the way we said it should be. Applied unmodified.
#
#   2. upstream-pr50514.patch   #50514 by yongqinwang-cmd, open since 2026-07-31.
#      Deletes the blanket "<method> with pipeline parallel is not supported" refusal
#      and replaces it with a per-model opt-in. It covers `dspark`, not just eagle3,
#      and it sets the opt-in on Kimi-K3 itself. Its author validated it on
#      Kimi-K3 + DSpark at TP8 x PP2 across two 8x B200 nodes, GSM8K 0.9644 +/- 0.0051
#      over the full 1,319-example set. That is our B200 target configuration.
#      Applied unmodified. This is why our dspark-pp-828 is not in this chain.
#
#   3. ours-k3-int64idx.patch   One line. No upstream PR covers it; searched
#      2026-08-31. `_store_cache_checkpoints_kernel` multiplies an int32 state_idx by
#      conv_state.stride(0) = 442,368, so the product passes 2**31 at
#      state_idx >= 4,855 and the store wraps negative. It presents as a CUDA illegal
#      memory access ~19.5 minutes in, long after health passes, which is why a 60
#      minute arm needs it and a smoke test does not. main already applies exactly
#      this fix, with the same reasoning, in mamba_utils.py:249 -- this kernel was
#      missed.
#
#   4. ours-mooncake-interleave.patch   One line on top of #54457. cjackal set the
#      new attribute False on the two CPU-offload connectors and not on
#      MooncakeStoreConnector, which is the same kind of thing: an offload tier that
#      moves no KV between instances and has handled DCP itself since #53324. Belongs
#      as a comment on #54457, not as a PR of its own.
#
#
# V2 ADDS ONE PATCH. v1 ran as pipeline 65516120 on 2026-08-31: the dspark7 arm
# completed, and the no-spec arm died in warmup with
#
#   mooncake/store/scheduler.py:424 in _apply_current_save_block_ids
#   AssertionError: Missing current block table for store request chatcmpl-...
#
# That is upstream #51358's, not ours -- `_apply_current_save_block_ids` does not
# exist in the 08/28 nightly and does exist in this one. The core snapshots block
# tables only for requests that are new this step, were allocated new blocks this
# step, or carry a boundary-state offload; the connector asserts an entry for every
# save-eligible request. A decode step that fills a block opened earlier satisfies
# neither, and the assert raises out of Scheduler.schedule(), killing EngineCore.
# DSpark allocates new blocks often enough to hide it, which is the whole of the
# arm split.
#
#   5. ours-mooncake-save-snapshot.patch  Declines that save instead of asserting.
#      Reproduced test-first: the new unit test fails on the same file, line and
#      message as the cluster crash and passes with the patch; 250 Mooncake store
#      tests pass with it. Not upstream and unreported as of 2026-08-31.
#
# ORDER MATTERS. #54457 first, because #50514 also edits config/vllm.py and applying
# the small one onto the clean tree keeps its hunks unambiguous. Ours go last so that
# dropping either is a one-line edit here when the corresponding PR merges.
#
# WHAT THIS CHAIN DOES NOT CARRY, deliberately: ppbuilders (untested structural fix,
# no reproducer), kdackpt-bounds (hypothesis for a fault int64idx explained), the KDA
# block-table width guard, the hybrid kv_load_failure hunk (its marker has to be
# re-located on this tree first), and every probe. If an arm fails in a way one of
# those would explain, that failure is the finding -- record it, do not silently add
# the patch back.
# =============================================================================
set -euo pipefail

readonly PINNED_SHA=44fe2a392b
readonly PATCH_DIR=/configs/patches

echo "=== kimi-k3-upstream-v2: upstream PRs only, on ${PINNED_SHA} ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[upstream-v2] vllm root: ${VLLM_ROOT}"

# --- base image check -------------------------------------------------------
# Not fatal: this chain is meant to be pointed at newer nightlies as they appear.
# But say loudly which image we are on, because every "skips" claim above is keyed
# to one tree and a newer one may have absorbed more.
python3 - <<PY
import re
import vllm

v = vllm.__version__
m = re.search(r"\+g([0-9a-f]+)", v)
if not m:
    print(f"[upstream-v2] WARNING: version {v!r} carries no +g<sha>; cannot verify base")
else:
    sha = m.group(1)
    pinned = "${PINNED_SHA}"
    if sha.startswith(pinned) or pinned.startswith(sha):
        print(f"[upstream-v2] base image confirmed: {sha}")
    else:
        print(f"[upstream-v2] WARNING: base is {sha}, this chain was derived against {pinned}")
        print("[upstream-v2] re-check the 'skips' list in this header before trusting a result")
PY

# --- apply ------------------------------------------------------------------
# Each patch is gated on a marker so a newer image that already carries it is a
# skip rather than a failure -- that skip IS the signal that upstream absorbed it.
apply_patch() {
    local name="$1" marker="$2" file="$3"
    if grep -q "${marker}" "${VLLM_ROOT}/${file}" 2>/dev/null; then
        echo "[upstream-v2] ${name}: already present in this image, skipping"
        return 0
    fi
    if ! patch -p1 -d "${VLLM_ROOT}" --dry-run --forward --fuzz=0 \
         < "${PATCH_DIR}/${name}" > "${TMPDIR:-/tmp}/${name}.dry" 2>&1; then
        echo "[upstream-v2] FATAL: ${name} does not apply to this image" >&2
        cat "${TMPDIR:-/tmp}/${name}.dry" >&2
        exit 1
    fi
    patch -p1 -d "${VLLM_ROOT}" --forward --fuzz=0 < "${PATCH_DIR}/${name}"
    echo "[upstream-v2] ${name}: applied"
}

apply_patch upstream-pr54457.patch \
    'requires_dcp_block_aligned_interleave' \
    vllm/distributed/kv_transfer/kv_connector/v1/base.py

apply_patch upstream-pr50514.patch \
    'supports_aux_hidden_states_over_pp' \
    vllm/models/kimi_k3/nvidia/model.py

apply_patch ours-k3-int64idx.patch \
    'seq_idx).to(tl.int64)' \
    vllm/models/kimi_k3/nvidia/kda.py

apply_patch ours-mooncake-interleave.patch \
    'requires_dcp_block_aligned_interleave = False' \
    vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py

apply_patch ours-mooncake-save-snapshot.patch \
    'no current block table for' \
    vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py

# --- verify -----------------------------------------------------------------
# Fail the job here rather than serve a half-patched engine. Each assertion names
# the property the arm depends on, not the patch that provides it, so this still
# passes if upstream lands the change in a different shape.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
rd = lambda p: open(os.path.join(root, p)).read()

failures = []

def want(rel, needle, why):
    if needle not in rd(rel):
        failures.append(f"{why}: {needle!r} missing from {rel}")

def forbid(rel, needle, why):
    if needle in rd(rel):
        failures.append(f"{why}: {needle!r} still present in {rel}")

# #50514 -- spec decode under PP must be reachable at all.
forbid("vllm/v1/worker/gpu/model_runner.py",
       "with pipeline parallel ",
       "PP+spec refusal was not lifted")
want("vllm/v1/worker/gpu/model_runner.py",
     "verify_supports_aux_hidden_states_over_pp",
     "PP+spec refusal was lifted without its replacement guard")
want("vllm/models/kimi_k3/nvidia/model.py",
     "supports_aux_hidden_states_over_pp = True",
     "K3 does not opt in to aux hidden states over PP")
want("vllm/models/kimi_k3/nvidia/model.py",
     "spec_decode_needs_target_embed",
     "the last PP stage will not load the target embedding")

# #54457 + ours -- the interleave must not be promoted for the Mooncake store.
want("vllm/config/vllm.py",
     "connector_cls.requires_dcp_block_aligned_interleave",
     "the DCP interleave promotion is still ungated")
want("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py",
     "requires_dcp_block_aligned_interleave = False",
     "MooncakeStoreConnector still requests block-aligned interleave")

# ours -- #51358's assert kills EngineCore on a save the core did not snapshot.
forbid("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py",
       "Missing current block table for store request",
       "the fatal Mooncake save assert is still present")
want("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py",
     "no current block table for",
     "the Mooncake save is not declined when the core has no snapshot")

# ours -- the int32 multiply that faults ~19.5 min in.
want("vllm/models/kimi_k3/nvidia/kda.py",
     "seq_idx).to(tl.int64)",
     "the KDA checkpoint state index is still int32")

if failures:
    for f in failures:
        print(f"[upstream-v2] FATAL: {f}", file=sys.stderr)
    sys.exit(1)

print("[upstream-v2] all 9 assertions passed")
PY

# Import the modules we edited, so a syntax or import error fails here and not in a
# worker 40 minutes into a sweep.
python3 - <<'PY'
import vllm.config.vllm  # noqa: F401
import vllm.distributed.kv_transfer.kv_connector.v1.base  # noqa: F401
import vllm.v1.worker.gpu.model_runner  # noqa: F401

print("[upstream-v2] edited modules import cleanly")
PY

echo "=== kimi-k3-upstream-v2: done ==="
