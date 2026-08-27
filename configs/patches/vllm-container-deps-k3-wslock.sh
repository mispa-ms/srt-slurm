#!/usr/bin/env bash
# Lock the workspace after capture on the V2 runner, as V1 already does.
# =============================================================================
# THE GAP. workspace.py refuses to grow once locked:
#
#   if current_size < required_bytes:
#       if self._locked:
#           raise AssertionError("Workspace growth is not allowed after locking.")
#       ... reallocate ...
#
# and lock_workspace() exists precisely for that -- its own docstring says "Lock after
# warmup/profiling". It is called once in the whole tree, from the V1 runner:
#
#   gpu_model_runner.py:7029   lock_workspace()
#     # Lock workspace to prevent resizing during execution.
#     # Max workspace sizes should have been captured during warmup/profiling.
#
# The V2 runner never calls it: zero occurrences anywhere under vllm/v1/worker/gpu/.
# DSpark forces V2 (config/vllm.py refuses to run it on V1), so this workstream has no
# choice but the unlocked path.
#
# WHY THAT MATTERS HERE. A grow reallocates, and breakable_cudagraph captures its eager
# segments with weak references on the stated assumption that "cudagraph owns the slot,
# so the weak_ref is safe to deref on replay". Workspace tensors are not in the
# cudagraph pool -- kda.py:866 takes checkpoint_state, final_state and workspace from
# current_workspace_manager().get_simultaneous(...) -- so a reallocation leaves a
# captured graph pointing at freed memory, and the replay faults. That is the illegal
# access, and it explains why seven index guards changed nothing: the indices were
# always in range, the base pointer was not.
#
# WHAT THIS ARM DOES, AND WHAT IT MEANS EITHER WAY. It mirrors V1: lock at the end of
# capture_model. This is a fix if the sizes really are settled by then, and a precise
# diagnostic if they are not -- the AssertionError names the caller and the two sizes,
# which is far better than a silent realloc followed by an illegal access thousands of
# requests later.
#
#   runs clean          the workspace was stable; growth was the fault
#   asserts at startup  something legitimately needs more after capture, and the
#                       message says who and how much -- that is the bug to fix
#   still faults        workspace growth is not the mechanism
#
# Run beside the wsprobe arm, which measures the same thing without changing behaviour.
# =============================================================================
set -euo pipefail

echo "=== wslock: lock the workspace after capture on the V2 runner ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_runner.py",
)
src = open(target).read()

if "[wslock]" in src:
    print("[wslock] already applied: " + target)
    sys.exit(0)

ANCHOR = """        end_time = time.perf_counter()
        end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[wslock] FATAL: expected one capture_model tail, found %d" % src.count(ANCHOR)
    )

ADDITION = '''        # [wslock] The V1 runner locks here -- "Max workspace sizes should have
        # been captured during warmup/profiling" -- and V2 never did. Unlocked, a
        # later grow reallocates the buffer that breakable_cudagraph's eager
        # segments still hold weak references to, and the replay dereferences freed
        # memory.
        from vllm.v1.worker.workspace import lock_workspace as _wslock_lock

        _wslock_lock()
        logger.info("[wslock] workspace locked after capture")

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
'''

patched = src.replace(ANCHOR, ADDITION, 1)
compile(patched, target, "exec")
open(target, "w").write(patched)
print("[wslock] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import inspect
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/v1/worker/gpu/model_runner.py")).read()
if src.count("[wslock]") < 2:
    sys.exit("[wslock] FATAL: marker missing after write")

from vllm.v1.worker.workspace import lock_workspace  # noqa: F401
import vllm.v1.worker.gpu.model_runner as mr

body = inspect.getsource(mr.GPUModelRunner.capture_model)
if "_wslock_lock()" not in body:
    sys.exit("[wslock] FATAL: the lock did not land inside capture_model")
print("[wslock] verified: lock_workspace importable and called from capture_model")
PY

echo "=== wslock: done ==="
