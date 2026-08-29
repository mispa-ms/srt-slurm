#!/usr/bin/env bash
# vLLM PR #51358 -- Mooncake must save the exact Mamba boundary states.
# =============================================================================
# WHAT BREAKS. In `align` mode the Mooncake save path resolved each group's source
# block positionally, from a worker-side snapshot of the request's block IDs. That is
# only valid while the block list is append-only, and it is not:
#
#   - superseded state positions are replaced by the null block and freed;
#   - speculative scratch blocks are relocated to different logical positions;
#   - the connector's block-ID mirror does not observe either change in place.
#
# So the store can persist null, stale/reassigned, or *live speculative* Mamba state
# under a valid prefix hash. Upstream calls this a correctness bug, and it is: a later
# reader gets a hash hit backed by the wrong state.
#
# WHY IT IS OUR CONFIGURATION EXACTLY. We run all three of the things this needs to go
# wrong at once -- `mamba-cache-mode align`, MooncakeStoreConnector, and speculative
# decoding (DSpark ns=4/ns=7 on every arm that sits on our frontier).
#
# WHY IT IS WORTH AN ARM. The trace is 99.3% input tokens, so y = P/(1-h)/N_gpu is
# effectively an identity and h is multiplicative on throughput. Measured on the c48
# ns=4 arm: 84.90% of input served from HBM, 15.10% falls through, Mooncake serves
# 55.73% of that, leaving a 6.68% true miss. A bug that makes a save unusable shows up
# as fall-through that Mooncake cannot answer -- which is exactly the 44% it misses.
#
# STATUS. Merged upstream 2026-08-29T02:40:54Z. Our 08/28 image was built at 06:13 UTC
# on 08/28, about 20 hours earlier, so it does not carry it.
#
# ORDERING. Four of its ten files are also edited by k3-mooncake-53324-828.patch
# (coordinator, data, scheduler, worker), so it must run after that patch. Dry-run on
# 6f7df92a+53324: clean, 0 rejects. It also has to run *before* #53614, which is cut
# against a tree that already contains it -- see vllm-container-deps-k3-pr53614.sh.
# =============================================================================
set -euo pipefail

echo "=== k3-pr51358: save exact Mamba boundary states ==="

VLLM_ROOT=${VLLM_ROOT:-$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')}
command -v patch >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq patch; }

cd "$VLLM_ROOT"

MC_SCHED=vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py

if grep -q "boundary_state_offloads" "$MC_SCHED" 2>/dev/null; then
  echo "[pr51358] already applied"
else
  # This patch is the difference between a correct and an incorrect store, so a partial
  # application is worse than no application: it would keep serving hash hits from state
  # the patch was meant to stop trusting. Dry-run first and let set -e kill the job.
  patch -p1 --forward --dry-run < /configs/patches/k3-pr51358-828.patch
  patch -p1 --forward < /configs/patches/k3-pr51358-828.patch
  echo "[pr51358] applied"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))

FILES = (
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    "vllm/v1/core/block_pool.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/sched/output.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
)
for rel in FILES:
    compile(open(os.path.join(root, rel)).read(), rel, "exec")

# Three named pieces of the fix, one per layer it touches. If the patch applied to a
# tree that had drifted, the hunks can land at the wrong offsets and still "succeed";
# these names are what the fix is actually made of.
WANT = {
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py": (
        "boundary_state_offloads",
    ),
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py": (
        "_boundary_snapshot_puts",
        "_maybe_offload_boundary_states",
    ),
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py": (
        "_build_aligned_boundary_store_jobs",
    ),
    "vllm/v1/core/kv_cache_manager.py": ("take_boundary_state_offloads",),
    "vllm/v1/core/block_pool.py": ("is_block_writable",),
}
for rel, syms in WANT.items():
    src = open(os.path.join(root, rel)).read()
    for s in syms:
        if s not in src:
            sys.exit(f"[pr51358] FATAL: {s} missing from {rel}")

# The relocation path is the reason positional resolution was unsafe. If it is not
# there, the patch did not bring the part that motivated it.
stkc = open(os.path.join(root, "vllm/v1/core/single_type_kv_cache_manager.py")).read()
if "_relocate_speculative_block" not in stkc:
    sys.exit("[pr51358] FATAL: _relocate_speculative_block missing")

import vllm.v1.core.block_pool  # noqa: F401
import vllm.v1.core.single_type_kv_cache_manager  # noqa: F401

print("[pr51358] verified: 10 files compile, boundary-state + relocation symbols present")
PY

echo "=== k3-pr51358: done ==="
