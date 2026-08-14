#!/usr/bin/env bash
# Round 2 of the vllm#51766 cost hunt: count what the fix actually changes,
# with the fix reverted.
#
# ROUND 1 IS WHY THIS EXISTS. The first hypothesis was that #51766's
# cached_blocks_this_step guard stalls the scheduler's waiting loop, because the
# refusal is delivered through the out-of-memory channel and the loop breaks on
# it. That was half right and useless: every waiting-loop allocation failure in
# an hour WAS a CoW deferral and none was memory pressure -- but there were
#
#     steps=39,500  alloc_fail=9  cow_break=9
#
# at c70. Nine stalls in forty thousand steps cannot be 8.9%, and a scheduler
# patch that skips instead of breaking moved the number by +1.5% at c70 and
# -0.3% at c48, both inside spread. So the guard is not the cost.
#
# WHAT IS LEFT. #51766 marks the request allocated on the no-new-block early
# return, which flips blocks_allocated to True on the NEXT partial-tail CoW.
# That routes it down the running-request branch, which -- unlike the
# first-prefill branch -- moves the cache entry to the CoW target, takes an extra
# ref on the source, and hands the block to the connector for partial-tail
# offload. That last one is a write to Mooncake, so it can be expensive at a
# frequency the deferral never reached.
#
# WHAT THIS PAIR MEASURES. Both arms log, every 500 scheduler steps:
#
#     cow-stats: steps=N early_return=N cow_running=N cow_firstprefill=N
#                cow_copies=N tail_offloads=N
#
# The fix arm should show cow_running and tail_offloads carrying the traffic; the
# reverted arm should show the same work arriving as cow_firstprefill with no
# tail_offloads. If cow_copies is the same on both and only tail_offloads
# differs, the extra Mooncake write is the candidate and the next patch has a
# target. If all six counters are small on both, the cost is not in this branch
# at all and the search moves out of MambaManager.
#
# THE REVERTED ARM IS A MEASUREMENT, NOT A CONFIGURATION. #51766 fixes real
# Mamba state corruption on this model -- 3/12 verbatim repeats in the author's
# pre-fix replay -- and nothing shipped will carry this revert.
set -euo pipefail

readonly PATCH=/configs/patches/k3-cowstats-rev.patch

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

src = (pathlib.Path(vllm.__file__).parent
       / "v1/core/single_type_kv_cache_manager.py").read_text()

for mark, who in [
    ('cow_stats = {', "the counter dict"),
    ('"cow-stats: steps=%d early_return=%d cow_running=%d "', "the log line"),
    ('MambaManager.cow_stats["cow_running"] += 1', "the running-branch counter"),
    ('MambaManager.cow_stats["cow_firstprefill"] += 1', "the first-prefill counter"),
    ('MambaManager.cow_stats["tail_offloads"] += 1', "the offload counter"),
]:
    assert mark in src, f"{who} is missing after the patch"

# The one line that separates the two arms, checked on the branch it belongs to
# rather than anywhere in the file.
cls = next(n for n in ast.walk(ast.parse(src))
           if isinstance(n, ast.ClassDef) and n.name == "MambaManager")
# ast.unparse re-parenthesises: the source reads `... and not has_partial_hit`
# but unparse emits `... and (not has_partial_hit)`, so match on the stable half.
guard = "num_required_blocks <= len(req_blocks)"
branches = [n for n in ast.walk(cls)
            if isinstance(n, ast.If) and guard in ast.unparse(n.test)]
assert len(branches) == 1, (
    f"expected one {guard!r} branch, found {len(branches)}"
)
body = ast.unparse(branches[0].body)
present = "_allocated_block_reqs.add" in body
assert present is False, (
    f"vllm#51766's line is {'present' if present else 'absent'} but this arm "
    f"needs it {'present' if False else 'absent'} -- the pair would compare "
    "a run to itself"
)
print("=== kimi-k3-nightly-v9-rev verified: counters on, #51766 reverted ===")
PYEOF
