#!/usr/bin/env bash
# Round 4 of the vllm#51766 cost hunt: split the engine step. #51766 reverted.
#
# WHERE THE HUNT STANDS. The fix costs 6.6% at c48 and 8.8% at c70 and it stays
# -- it prevents Mamba state corruption on this model. Three rounds have each
# ruled something out rather than found the cause:
#
#   R1  the same-step CoW deferral fires 9 times in 39,500 scheduler steps, and
#       skipping instead of breaking the waiting loop moved +1.5% / -0.3%.
#   R2  every event on both CoW branches counted with the fix present and
#       reverted: per step all six counters match, 56% vs 55% branch split,
#       tail_offloads tracking cow_running exactly on both.
#   R3  iteration logs localised it -- DECODE steps are 36.81 ms against 27.17
#       (+35%) while PREFILL is 309.1 against 308.5 (+0.2%), same batch size and
#       same tokens per step. The weighted step ratio is 1.102, which matches
#       R2's "the reverted engine completes 11.6% more steps per hour" exactly.
#
# WHY NOT ANOTHER TRACE. R3 also took two 30 s nsys captures and they could not
# answer it. The two windows had different prefill/decode mixes (fix 11.1 s of
# prefill in-window against rev 22.0 s), which is enough to change GEMM tile
# selection and allreduce specialisation, so a kernel-level diff of the two is
# mostly a diff of what each window happened to be doing. Iteration mode fixes
# the number of steps captured, not their composition, and these traces carry no
# vLLM iteration NVTX to slice by -- only nccl markers. Isolating decode by the
# worker log fails too: the log has 1 s resolution and prefill steps are
# 250-500 ms, so a "pure decode second" is not pure. On the reverted arm only
# 56% of such a second was accounted for by logged decode iterations.
#
# WHAT THIS ROUND ASKS. One question with a decisive answer: is the extra time
# inside schedule() or outside it? schedule() is the whole host-side scheduler,
# including every MambaManager call #51766 touches. Every 500 steps:
#
#     step-split: steps=N sched_total_ms=N sched_per_call_us=N early_return=N
#
#   sched_per_call_us differs  -> the cost is host scheduling, and the next step
#                                 is a Python profile of schedule() alone.
#   sched_per_call_us matches  -> the cost is in the worker: forward, block
#                                 tables, or the copy stream. A trace is then
#                                 worth taking, and it will know what to look at.
#
# The timer is two perf_counter_ns calls per step against a step of 27-37 ms, so
# it cannot move what it measures.
#
# THE REVERTED ARM IS A MEASUREMENT. Nothing shipped carries this revert.
set -euo pipefail

readonly PATCH=/configs/patches/k3-stepsplit-rev.patch

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-nightly-v6.sh"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")

echo "=== applying $(basename "$PATCH") ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH"
fi

python3 - <<'PYEOF'
import ast
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent
sched = (root / "v1/core/sched/scheduler.py").read_text()
mgr = (root / "v1/core/single_type_kv_cache_manager.py").read_text()

for src, mark, who in [
    (sched, "def _schedule_body(", "the renamed body"),
    (sched, '"step-split: steps=%d sched_total_ms=%.1f sched_per_call_us=%.1f "',
     "the log line"),
    (mgr, "MambaManager.n_early += 1", "the early-return counter"),
]:
    assert mark in src, f"{who} is missing after the patch"

# The wrapper must call the body, or the timer measures nothing and the engine
# schedules nothing.
tree = ast.parse(sched)
cls = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
wrap = next(n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "schedule")
body = ast.unparse(wrap)
assert "self._schedule_body(throttle_prefills)" in body, (
    "schedule() does not call _schedule_body; the rename did not take"
)
assert "perf_counter_ns" in body, "schedule() is not timed"

# The one line that separates the arms, checked on its own branch.
mcls = next(n for n in ast.walk(ast.parse(mgr))
            if isinstance(n, ast.ClassDef) and n.name == "MambaManager")
guard = "num_required_blocks <= len(req_blocks)"
branches = [n for n in ast.walk(mcls)
            if isinstance(n, ast.If) and guard in ast.unparse(n.test)]
assert len(branches) == 1, f"expected one guard branch, found {len(branches)}"
present = "_allocated_block_reqs.add" in ast.unparse(branches[0].body)
assert present is False, (
    f"vllm#51766 is {'present' if present else 'absent'} but this arm needs it "
    f"{'present' if False else 'absent'} -- the pair would compare a run to itself"
)
print("=== kimi-k3-nightly-v11-rev verified: step split on, #51766 reverted ===")
PYEOF
