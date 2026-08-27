#!/usr/bin/env bash
# Include checkpoint_state in the checkpoint kernel's grid bound, and report the gap.
# =============================================================================
# WHAT THE PROBE FOUND, AND WHAT IT DID NOT. The index probe reported
#
#   token_idx [-3,7167] vs x_rows=8192 | state_idx [0,22] vs conv=5628 recur=5628
#   state_len=3 | off[min,max]=[0,6144] | qsl[min,max]=[0,1024]
#
# The upper bounds are all comfortable. The -3 is real arithmetic -- a row with
# checkpoint_offset 0 gives token_idx = 0 + 0 - STATE_LEN + history -- but the kernel
# already excludes those rows: valid_conv carries `checkpoint_offset >= STATE_LEN`.
# So the conv half is guarded and the probe's lower-bound flag is benign. Reporting it
# as the bug would have been the fifth wrong fix in a row.
#
# What the probe never looked at is the recurrent half:
#
#   valid_recurrent = (cols < RECURRENT_ROW_SIZE) & valid_checkpoint
#   recurrent = tl.load(recurrent_checkpoint_ptr + seq_idx * checkpoint_stride_0 + cols,
#                       mask=valid_recurrent)
#
# recurrent_checkpoint_ptr is `checkpoint_state`, indexed by seq_idx, and nothing bounds
# seq_idx against checkpoint_state.shape[0]. valid_checkpoint only tests state_idx and
# checkpoint_offset; valid_recurrent only tests cols. The grid is
# checkpoint_offsets.numel().
#
# This also explains why the earlier grid clamp failed. That arm took
#
#   min(checkpoint_offsets, checkpoint.state_indices, non_spec_query_start_loc)
#
# -- three tensors, and checkpoint_state was not one of them. Its mismatch logger never
# fired because those three genuinely do agree. The fourth was never checked.
#
# THE CHANGE. Add checkpoint_state.shape[0] to the bound, and log the shapes whenever
# it is the binding one, so the run says plainly whether this was it. Instrumented, not
# asserted: four theories have already died from being neither.
#
# When every tensor agrees, the grid is unchanged and behaviour is identical.
# =============================================================================
set -euo pipefail

echo "=== ckptstate-bounds: bound the checkpoint grid by checkpoint_state too ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda.py",
)
src = open(target).read()

if "[ckptstate]" in src:
    print("[ckptstate] already applied: " + target)
    sys.exit(0)

ANCHOR = """                        block_size = 256
                        _store_cache_checkpoints_kernel[
                            (
                                checkpoint_offsets.numel(),
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[ckptstate] FATAL: expected one unclamped checkpoint launch, found %d; either "
        "the call moved or another patch already rewrote the grid"
        % src.count(ANCHOR)
    )

ADDITION = '''                        block_size = 256
                        # [ckptstate] The kernel reads checkpoint_state at
                        # seq_idx * checkpoint_stride_0 with no bound on seq_idx, while
                        # the grid comes from checkpoint_offsets. Bound it by every
                        # tensor the kernel indexes by seq_idx, checkpoint_state
                        # included -- the one the earlier three-tensor clamp missed.
                        _cks_n = min(
                            checkpoint_offsets.numel(),
                            checkpoint.state_indices.numel(),
                            non_spec_query_start_loc.numel(),
                            checkpoint_state.shape[0],
                        )
                        if _cks_n != checkpoint_offsets.numel():
                            logger.warning(
                                "[ckptstate] grid %d -> %d | offsets=%d "
                                "state_indices=%d qsl=%d checkpoint_state=%s",
                                checkpoint_offsets.numel(),
                                _cks_n,
                                checkpoint_offsets.numel(),
                                checkpoint.state_indices.numel(),
                                non_spec_query_start_loc.numel(),
                                tuple(checkpoint_state.shape),
                            )
                        _store_cache_checkpoints_kernel[
                            (
                                _cks_n,
'''

patched = src.replace(ANCHOR, ADDITION, 1)
compile(patched, target, "exec")
open(target, "w").write(patched)
print("[ckptstate] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
if src.count("[ckptstate]") < 2:
    sys.exit("[ckptstate] FATAL: marker missing after write")
if "_cks_n," not in src:
    sys.exit("[ckptstate] FATAL: the grid still uses the unclamped count")
if "checkpoint_state.shape[0]," not in src:
    sys.exit("[ckptstate] FATAL: checkpoint_state is not in the bound")

import vllm.models.kimi_k3.nvidia.kda as kda  # noqa: F401

if "logger" not in dir(kda):
    sys.exit("[ckptstate] FATAL: kda module has no logger")
print("[ckptstate] verified: module imports, grid bounded by four tensors")
PY

echo "=== ckptstate-bounds: done ==="
