#!/usr/bin/env bash
# Report every workspace reallocation, with the pointer that just died.
# =============================================================================
# WHY THIS AND NOT AN EIGHTH BOUNDS FIX. Seven attempts guarded an index. All seven
# failed, and the probe that measured the indices found every one of them inside its
# tensor. The indices were never the problem.
#
# What the launch-blocking traceback actually says, read properly this time:
#
#   breakable_cudagraph.py:214  replay
#   breakable_cudagraph.py:115  return capture.add_eager(
#                                   lambda: fn(*weak_args, **weak_kwargs))
#   kda.py:904                  _store_cache_checkpoints_kernel[...]
#   RuntimeError: Triton Error [CUDA]: an illegal memory access
#
# The kernel runs as an eager segment inside a CUDA graph replay, against arguments
# captured earlier as *weak references*. The comment above that line states the
# assumption that makes it safe:
#
#   # Weak-ref args: ... cudagraph owns the slot, so the weak_ref is safe to deref
#   # on replay.
#
# That holds for tensors in the cudagraph pool. checkpoint_state, final_state and
# workspace are not: kda.py:866 takes them from
# current_workspace_manager().get_simultaneous(...). If the WorkspaceManager grows, it
# reallocates, and every earlier view -- including the ones a captured graph still
# points at -- is dead memory.
#
# AND NOTHING PREVENTS THE GROWTH ON OUR PATH. workspace.py guards it:
#
#   if current_size < required_bytes:
#       if self._locked:
#           raise AssertionError("Workspace growth is not allowed after locking.")
#       ... reallocate ...
#
# and lock_workspace() is called exactly once in the tree, from the V1 runner
# (gpu_model_runner.py:7029, "Lock after warmup/profiling"). The V2 runner never calls
# it -- zero occurrences under vllm/v1/worker/gpu/. DSpark forces V2, so we are on the
# unlocked path with no choice.
#
# This fits every observation the seven bounds fixes could not: indices in range but
# the base pointer stale; the fault landing in a graph replay's eager segment; a
# different kernel each run; PP-only, because two microbatches of different sizes make
# a mid-serving grow far more likely; and PP-off running 3628 s clean, its size settled
# after warmup.
#
# WHAT THIS ARM DOES. It measures, and changes nothing else. On every actual
# reallocation it logs the old pointer and size, the new pointer and size, the slot,
# and the caller -- workspace.py already computes get_caller_info() for its assertion
# message, so the attribution is free.
#
#   reallocations after warmup  -> the weak refs are dangling and this is the bug
#   none                        -> the workspace is stable and this theory joins the
#                                  other seven
# =============================================================================
set -euo pipefail

echo "=== wsprobe: log workspace reallocations ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/workspace.py",
)
src = open(target).read()

if "[wsprobe]" in src:
    print("[wsprobe] already applied: " + target)
    sys.exit(0)

# The reallocation happens right after the locked-check. Anchor on the comment that
# introduces it so we sit before the new buffer replaces the old one.
ANCHOR = """            # Only resize the requesting ubatch/lane workspace. Other slots
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[wsprobe] FATAL: expected one resize comment, found %d; workspace.py moved"
        % src.count(ANCHOR)
    )

ADDITION = '''            # [wsprobe] Diagnostic. A grow here reallocates, and any weak
            # reference a captured cudagraph still holds to the old buffer becomes a
            # dangling pointer. Record what died.
            if _WSPROBE["n"] < 60:
                _WSPROBE["n"] += 1
                logger.warning(
                    "[wsprobe] #%d slot=%d grow %.2f -> %.2f MB | old_ptr=%s | "
                    "caller=%s",
                    _WSPROBE["n"],
                    workspace_id,
                    current_size / _MB,
                    required_bytes / _MB,
                    hex(current_workspace.data_ptr())
                    if current_workspace is not None
                    else "none",
                    get_caller_info(),
                )
            # Only resize the requesting ubatch/lane workspace. Other slots
'''

src = src.replace(ANCHOR, ADDITION, 1)

COUNTER_ANCHOR = "logger = init_logger(__name__)\n"
if src.count(COUNTER_ANCHOR) != 1:
    sys.exit(
        "[wsprobe] FATAL: expected one module logger, found %d"
        % src.count(COUNTER_ANCHOR)
    )
src = src.replace(
    COUNTER_ANCHOR,
    COUNTER_ANCHOR + '\n# [wsprobe] report cap\n_WSPROBE = {"n": 0}\n',
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[wsprobe] applied: " + target)
PY

# get_caller_info is defined inside the grow branch, above where we log; confirm the
# module still imports and the names the probe uses are really in scope.
python3 - <<'PY'
import importlib.util
import inspect
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/v1/worker/workspace.py")).read()
if src.count("[wsprobe]") < 3:
    sys.exit("[wsprobe] FATAL: markers missing after write")

import vllm.v1.worker.workspace as ws

for name in ("logger", "_WSPROBE", "_MB", "WorkspaceManager"):
    if name not in dir(ws):
        sys.exit("[wsprobe] FATAL: %s missing from workspace module" % name)

body = inspect.getsource(ws.WorkspaceManager._ensure_workspace_size)
if body.index("def get_caller_info") > body.index("[wsprobe]"):
    sys.exit("[wsprobe] FATAL: the probe logs before get_caller_info is defined")
print("[wsprobe] verified: module imports, probe sits after get_caller_info")
PY

echo "=== wsprobe: done ==="
