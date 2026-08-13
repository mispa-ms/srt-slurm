#!/usr/bin/env bash
# v8-skip: the instrumentation arm plus the one behaviour change it is meant to
# price. Treatment arm; kimi-k3-nightly-v8-instr.sh is the control.
#
# THE CHANGE, IN FULL. In the scheduler's waiting loop, when allocate_slots
# returns None and the coordinator says the refusal was a CoW copy queued this
# step rather than a lack of blocks, skip that one request instead of breaking
# the loop:
#
#     request_queue.pop_request()
#     step_skipped_waiting.prepend_request(request)
#     continue
#
# That is not a new mechanism. The same loop already does exactly this for
# WAITING_FOR_REMOTE_KVS, for stale output still in flight, and for max_loras,
# and the running loop carries a note saying a continue instead of a break is
# accepted here at the cost of strict FCFS.
#
# WHAT IT DOES NOT CHANGE. The deferred request still does not run this step,
# which is the whole point of vllm#51766's guard -- the copy has not happened, so
# its block may not be read. Nothing about Mamba state ownership moves. What
# changes is only that the requests QUEUED BEHIND it, which have no interest in
# that block, are no longer held back with it.
#
# WHAT WOULD FALSIFY THE HYPOTHESIS. If cow_break is a small share of alloc_fail
# in the control, the stalls are memory pressure and this arm should read the
# same as the control. If the share is large and this arm still reads the same,
# the deferral is not what costs the 8.9% and the search continues elsewhere.
#
# WHAT SUCCESS LOOKS LIKE. Approaching the #51766-reverted arm's 12,912 at c70
# while keeping the fix. That combination is the point: the revert is not
# shippable, this is.
set -euo pipefail

readonly INSTR=/configs/patches/k3-cowdefer-instr.patch
readonly SKIP=/configs/patches/k3-cowdefer-skip.patch

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-nightly-v6.sh"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")

for p in "$INSTR" "$SKIP"; do
    echo "=== applying $(basename "$p") ==="
    if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$p" >/dev/null 2>&1; then
        echo "already applied"
    else
        patch -p1 --forward -d "$SITE" < "$p"
    fi
done

python3 - <<'PY'
import ast
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent
sched = (root / "v1/core/sched/scheduler.py").read_text()
mgr = (root / "v1/core/single_type_kv_cache_manager.py").read_text()

for src, mark, who in [
    (mgr, "MambaManager.cow_defer_events += 1", "the deferral counter"),
    (sched, '"cow-defer: steps=%d alloc_fail=%d cow_break=%d defer_calls=%d"',
     "the log line"),
]:
    assert mark in src, f"{who} is missing after the patches"

# The skip itself. Prove it is inside the deferral branch rather than anywhere in
# the file: walk to the `if ...deferred_for_pending_cow:` and require the skip
# calls in its body. A string search would pass on the loop's other skip sites.
tree = ast.parse(sched)
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "schedule")
branches = [n for n in ast.walk(fn)
            if isinstance(n, ast.If) and "deferred_for_pending_cow" in ast.unparse(n.test)]
assert len(branches) == 1, (
    f"expected one deferred_for_pending_cow branch in schedule(), found "
    f"{len(branches)}"
)
body = ast.unparse(branches[0])
for mark in ("request_queue.pop_request()",
             "step_skipped_waiting.prepend_request(request)",
             "continue"):
    assert mark in body, (
        f"{mark!r} is not inside the deferral branch; this arm is the control "
        "with extra counters and the A/B would compare a run to itself"
    )
print("=== k3-cowdefer-skip verified (skip one request, do not break the loop) ===")
PY
