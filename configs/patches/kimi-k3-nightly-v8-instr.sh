#!/usr/bin/env bash
# v8-instr: v6 plus instrumentation for the same-step CoW deferral. No behaviour
# change -- this arm exists to find out whether the mechanism we think costs 8.9%
# actually fires, before spending a run on a fix for it.
#
# WHAT WE THINK IS HAPPENING. vllm#51766 is a correctness fix for this model and
# it stays; see the AGG v5 Bisect tab. Its running-request CoW branch does one
# thing the first-prefill branch does not:
#
#     self.cached_blocks_this_step.add(cow_block.block_hash)
#
# which is right -- the copy has not run, so nobody may read that block this
# step. The problem is how the refusal is delivered. MambaManager.
# get_num_blocks_to_allocate returns num_gpu_blocks + 1, allocate_slots returns
# None, and the scheduler's waiting loop reads that as out of memory and BREAKS.
# One request waiting on a queued copy therefore stops every request behind it
# for the step, however many free blocks there are.
#
# WHY THIS WORKLOAD PAYS FOR IT. AgentX replays shared conversation prefixes, so
# many requests want the same hash, and Mooncake supplies external hits
# continuously. 69 of 93 layers are KDA. The result is that the branch is our
# common path rather than a corner.
#
# WHAT THIS ARM MEASURES. Every 500 scheduler steps:
#
#     cow-defer: steps=N alloc_fail=N cow_break=N defer_calls=N
#
# alloc_fail is every waiting-loop allocation failure; cow_break is the subset
# where the coordinator says the refusal was a pending copy rather than a lack of
# blocks. cow_break/alloc_fail is the share of queue stalls that are not memory
# pressure. If that share is small, the hypothesis is wrong and the skip arm is
# not worth running.
#
# The counters are three integer increments per failed allocation and one log
# line per 500 steps, so they do not move the number they are measuring.
set -euo pipefail

readonly PATCH=/configs/patches/k3-cowdefer-instr.patch

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-nightly-v6.sh"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")

echo "=== applying k3-cowdefer-instr ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH"
fi

python3 - <<'PY'
import ast
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent
sched = (root / "v1/core/sched/scheduler.py").read_text()
mgr = (root / "v1/core/single_type_kv_cache_manager.py").read_text()
coord = (root / "v1/core/kv_cache_coordinator.py").read_text()

for src, mark, who in [
    (mgr, "MambaManager.cow_defer_events += 1", "the deferral counter"),
    (mgr, "self.deferred_for_pending_cow = True", "the deferral flag"),
    (coord, "m.deferred_for_pending_cow for m in self.single_type_managers",
     "the coordinator aggregation"),
    (sched, "self._cow_break_events += 1", "the scheduler counter"),
    (sched, '"cow-defer: steps=%d alloc_fail=%d cow_break=%d defer_calls=%d"',
     "the log line"),
]:
    assert mark in src, f"{who} is missing after the patch"

# This arm must NOT change scheduling. Walk to the deferral branch and require
# that it only counts -- if the skip patch ever leaked in here, the two arms
# would be the same run and the A/B would prove nothing.
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
leaked = [m for m in ("step_skipped_waiting.prepend_request", "request_queue.pop_request")
          if m in body]
assert not leaked, (
    f"the skip patch leaked into the control arm: {leaked}. This A/B would "
    "compare a run to itself."
)
print("=== k3-cowdefer-instr verified (counters only, no behaviour change) ===")
PY
