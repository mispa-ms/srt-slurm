#!/usr/bin/env bash
# #53614's two asserts become the return None its own function already uses.
# =============================================================================
# WHAT HAPPENED. #53614 applied and verified, and the server died five minutes into
# serving -- long before the IMA window, which opens 19.5 minutes after the first
# forward. So that arm said nothing about the illegal access:
#
#   scheduler.py:667              schedule
#   kv_cache_manager.py:566       allocate_slots
#   kv_cache_coordinator.py:751   cache_blocks
#   single_type_kv_cache_manager.py:1789  _cache_partial_tail_block
#       assert not checkpoint_block.is_null
#
# That assert is added by #53614; it is not in the nightly. The PR is unfinished against
# this configuration -- prefix-match-unit 128 with a Mooncake external tier -- and a null
# block reaches that slot during ordinary serving.
#
# THE REPAIR IS THE FUNCTION'S OWN. Twenty lines below, the second branch of the same
# function treats exactly these two conditions as ordinary and returns:
#
#     if block_idx >= len(blocks):
#         return None
#     source_block = blocks[block_idx]
#     if source_block.is_null:
#         return None
#
# So the first branch is made to match. This is not a guess about intent: skipping the
# checkpoint export when there is no block to export into is what the rest of the
# function does, and it is what the pre-#53614 code did by never looking there at all.
#
# AND IT COUNTS, WHICH IS THE POINT. A guard that fires on every request would turn the
# patch into a no-op and make a clean run prove nothing -- the same trap as an arm that
# ends early reading as a fix. The counters log at 1, 10, 100, 1000, ... occurrences, so
# the log says how often the new path was actually taken:
#
#   [nullguard] rarely or never fires  -> the new hash-block offset ran, and the IMA
#                                         result means something
#   [nullguard] fires constantly       -> #53614 is disabled in practice here; a clean
#                                         run proves nothing and the real question is
#                                         why the checkpoint block is null
#
# Chained after #53614, so the only difference against that arm is these two lines.
# =============================================================================
set -euo pipefail

echo "=== pr53614-nullguard: skip the export instead of asserting ==="

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
target = os.path.join(root, "vllm/v1/core/single_type_kv_cache_manager.py")
src = open(target).read()

if "[nullguard]" in src:
    print("[nullguard] already applied")
    sys.exit(0)

ANCHOR = """            checkpoint_idx = cdiv(num_tokens, self.block_size) - 2
            blocks = self.req_to_blocks[request.request_id]
            assert 0 <= checkpoint_idx < len(blocks)
            checkpoint_block = blocks[checkpoint_idx]
            assert not checkpoint_block.is_null
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[nullguard] FATAL: expected one #53614 assert pair, found %d -- is #53614 "
        "applied, and is this the version it was written against?" % src.count(ANCHOR)
    )

ADDITION = """            checkpoint_idx = cdiv(num_tokens, self.block_size) - 2
            blocks = self.req_to_blocks[request.request_id]
            # [nullguard] #53614 asserts here; the second branch of this same function
            # returns for both conditions. Match it, and count how often it happens --
            # a guard that always fires would make the patch a no-op.
            if not (0 <= checkpoint_idx < len(blocks)):
                _NULLGUARD["range"] += 1
                _nullguard_report("range", _NULLGUARD["range"])
                return None
            checkpoint_block = blocks[checkpoint_idx]
            if checkpoint_block.is_null:
                _NULLGUARD["null"] += 1
                _nullguard_report("null", _NULLGUARD["null"])
                return None
            _NULLGUARD["ok"] += 1
            _nullguard_report("exported", _NULLGUARD["ok"])
"""

src = src.replace(ANCHOR, ADDITION, 1)

# Module state and a reporter that is bounded by value, not by a call cap.
# This file has no module logger of its own -- it is a pure data-structure module -- so
# the state block brings one. Two probes on this bug were written against modules that
# turned out to have no `logger` and only failed at import time on the cluster.
ANCHOR2 = "class SingleTypeKVCacheManager(ABC):"
if src.count(ANCHOR2) != 1:
    sys.exit("[nullguard] FATAL: expected one SingleTypeKVCacheManager, found %d" % src.count(ANCHOR2))
if "init_logger" not in src:
    src = src.replace(ANCHOR2, "from vllm.logger import init_logger\n\n" + ANCHOR2, 1)
src = src.replace(
    ANCHOR2,
    '''logger = init_logger(__name__)

# [nullguard] how often #53614's checkpoint export is skipped, and how often it happens
_NULLGUARD = {"range": 0, "null": 0, "ok": 0}


def _nullguard_report(kind: str, n: int) -> None:
    """Log at 1, 10, 100, 1000, then every 10000 -- a cadence set by the counts
    themselves, so it cannot go quiet the way a per-worker report cap does."""
    if n in (1, 10, 100, 1000) or n % 10000 == 0:
        logger.warning(
            "[nullguard] %s=%d | skipped_range=%d skipped_null=%d exported=%d",
            kind, n, _NULLGUARD["range"], _NULLGUARD["null"], _NULLGUARD["ok"],
        )


''' + ANCHOR2,
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[nullguard] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/v1/core/single_type_kv_cache_manager.py")).read()

if "assert not checkpoint_block.is_null" in src:
    sys.exit("[nullguard] FATAL: the assert that killed the last arm is still there")
# Three: the inline comment at the guard, the state comment, and the log format. The
# substantive invariants are the four checks below; this only catches a partial write.
if src.count("[nullguard]") < 3:
    sys.exit("[nullguard] FATAL: markers missing after write (%d)" % src.count("[nullguard]"))
for need in ("_NULLGUARD", "_nullguard_report", "if checkpoint_block.is_null:"):
    if need not in src:
        sys.exit("[nullguard] FATAL: %s missing" % need)

import vllm.v1.core.single_type_kv_cache_manager as m

if m._NULLGUARD != {"range": 0, "null": 0, "ok": 0}:
    sys.exit("[nullguard] FATAL: counter state is not fresh")
m._nullguard_report("selftest", 1)
print("[nullguard] verified: assert gone, counters fresh, reporter callable")
PY

echo "=== pr53614-nullguard: done ==="
