#!/usr/bin/env bash
# Keep _store_cache_checkpoints_kernel inside all three tensors it indexes.
# =============================================================================
# THE FAULT. CUDA_LAUNCH_BLOCKING pinned the PP2 illegal memory access to one launch:
#
#   vllm/models/kimi_k3/nvidia/kda.py:904  _store_cache_checkpoints_kernel[...]
#   RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
#
# Four earlier arms blamed something else each time, because CUDA reports asynchronous
# errors at the next synchronising call: the tracebacks named an H2D copy, a whole
# cudagraph replay, and gpu_worker.execute_model. With launches serialised the error
# surfaced where it is caused, and the candidate list collapsed to this kernel.
#
# The kernel is new. It appears twice in the current nightly (a9a17e70) and zero times
# in the image our published numbers come from (728d3ad), which is consistent with PP2
# being fine there and faulting here.
#
# THE BUG. The grid is sized from one tensor and the kernel indexes three:
#
#   grid dim 0 = checkpoint_offsets.numel()
#
#   seq_idx = tl.program_id(0)
#   state_idx      = tl.load(checkpoint_state_indices_ptr + seq_idx)        # unmasked
#   checkpoint_off = tl.load(checkpoint_offsets_ptr + seq_idx * stride)
#   checkpoint_end = tl.load(query_start_loc_ptr + seq_idx) + checkpoint_off # unmasked
#
# Every later load and store carries mask=valid_conv or mask=valid_recurrent. These two
# do not. If checkpoint_offsets is longer than checkpoint.state_indices or than
# non_spec_query_start_loc, the extra programs read past the end of those tensors.
#
# WHY PIPELINE PARALLELISM. PP keeps pp_size microbatches in flight, each with its own
# non_spec_query_start_loc and its own request count, so the three lengths have room to
# disagree. With PP off there is one batch and they agree -- which is exactly what we
# measured: 3628 s and 2231 requests with zero illegal accesses on PP off, against four
# faulting PP2 arms.
#
# THE CHANGE. Clamp the grid to the shortest of the three tensors, at the call site.
# No kernel signature change, so no new Triton compile variants and no risk of an
# argument-order mistake -- the call passes twenty arguments positionally. When the
# three lengths agree, which is the healthy case and every non-PP case, the grid is
# unchanged and behaviour is bit-identical. When they disagree, the programs that would
# have read out of bounds simply do not launch; they could not have stored anything
# useful, since they had no state slot to write to.
#
# It also logs the lengths once when they differ, so the run tells us whether the
# mismatch is real and how large -- this is as much a measurement as a fix. If the
# clamp stops the fault and the log shows a mismatch, the mechanism is confirmed and
# this is worth reporting upstream.
#
# NOT A NO-OP TO VERIFY BY EYE: the marker check below re-reads the file and the module
# is imported, because a silently-skipped edit here would look exactly like "the fix
# did not work".
# =============================================================================
set -euo pipefail

echo "=== kdackpt-bounds: clamp the KDA checkpoint grid to its shortest tensor ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda.py",
)
src = open(target).read()

if "[kdackpt]" in src:
    print("[kdackpt] already applied: " + target)
    sys.exit(0)

ANCHOR = """                        block_size = 256
                        _store_cache_checkpoints_kernel[
                            (
                                checkpoint_offsets.numel(),
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[kdackpt] FATAL: expected one _store_cache_checkpoints_kernel launch, found "
        "%d; the call moved and this needs re-deriving" % src.count(ANCHOR)
    )

ADDITION = """                        block_size = 256
                        # [kdackpt] The grid was sized from checkpoint_offsets alone,
                        # but the kernel also indexes checkpoint.state_indices and
                        # query_start_loc unmasked. Under PP those lengths can differ
                        # and the extra programs read out of bounds. Launch only what
                        # all three cover.
                        _kdackpt_n = min(
                            checkpoint_offsets.numel(),
                            checkpoint.state_indices.numel(),
                            non_spec_query_start_loc.numel(),
                        )
                        if _kdackpt_n != checkpoint_offsets.numel():
                            logger.warning_once(
                                "[kdackpt] length mismatch: offsets=%d "
                                "state_indices=%d query_start_loc=%d -> grid %d",
                                checkpoint_offsets.numel(),
                                checkpoint.state_indices.numel(),
                                non_spec_query_start_loc.numel(),
                                _kdackpt_n,
                            )
                        _store_cache_checkpoints_kernel[
                            (
                                _kdackpt_n,
"""

patched = src.replace(ANCHOR, ADDITION, 1)
compile(patched, target, "exec")
open(target, "w").write(patched)
print("[kdackpt] applied: " + target)
PY

# The module must still import, and the marker must be there. A skipped or broken edit
# would be indistinguishable from "the fix did not help" once the run is over.
python3 - <<'PY'
import importlib.util
import inspect
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
if src.count("[kdackpt]") < 2:
    sys.exit("[kdackpt] FATAL: marker missing after write")
if "_kdackpt_n," not in src:
    sys.exit("[kdackpt] FATAL: the grid still uses the unclamped count")

import vllm.models.kimi_k3.nvidia.kda as kda  # noqa: F401

if "logger" not in dir(kda):
    sys.exit("[kdackpt] FATAL: kda module has no logger; warning_once would NameError")
print("[kdackpt] verified: module imports, grid clamped, logger present")
PY

echo "=== kdackpt-bounds: done ==="
