#!/usr/bin/env bash
# Bound checkpoint_cols by the block table's width, not just by zero.
# =============================================================================
# WHERE THIS CAME FROM. CUDA_LAUNCH_BLOCKING named the faulting launch:
#
#   kimi_k3/nvidia/kda.py:904  _store_cache_checkpoints_kernel
#   RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
#
# The obvious reading -- that the grid outruns the tensors it indexes -- was tested and
# is wrong: clamping the grid to the shortest of the three changed nothing, and the
# mismatch logger it carried never fired once. The lengths agree. So the bad index is a
# *value*, not a count.
#
# THE VALUE. The kernel's very first load is
#
#   state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx)
#
# and state_idx then addresses conv_state and recurrent_state directly:
#
#   conv_state_ptr + state_idx * state_stride_0 + ...
#
# A garbage state_idx walks straight off those tensors. And state_idx is built in
# kda_metadata.py like this:
#
#   checkpoint_cols.append(seq_len // block_size - 1 if valid else -1)
#   checkpoint_state_indices = m.block_table_tensor[request_rows_tensor,
#                                                  checkpoint_cols_tensor]
#   checkpoint_state_indices = torch.where(checkpoint_cols_tensor >= 0,
#                                          checkpoint_state_indices, NULL_BLOCK_ID)
#
# The guard checks the lower bound only. Nothing checks checkpoint_cols against
# block_table_tensor.size(1). And seq_len here is m.seq_lens_cpu_upper_bound -- an
# upper bound, by its own name -- so `seq_len // block_size - 1` can exceed the width
# the block table actually has.
#
# TWO SYMPTOMS, ONE CAUSE. That indexing is a torch advanced index, which runs
# IndexKernel. The ns7 arm on this branch, once its startup deadlock was fixed, died
# with exactly that:
#
#   IndexKernel.cu:111 Assertion `-sizes[i] <= index && index < sizes[i]
#                                 && "index out of bounds"` failed
#
# When the read is caught, you get that assert. When it is not -- the block table is
# small, and reading a little past it can land on mapped memory -- you get a plausible
# but wrong state_idx, and the fault moves downstream into the Triton kernel that
# dereferences it. That is the illegal access we have been chasing, and it is why the
# two symptoms never appeared together.
#
# WHY PIPELINE PARALLELISM. seq_lens_cpu_upper_bound is per microbatch. PP keeps
# pp_size microbatches in flight, so the bound can run ahead of the block table row it
# is paired with. With PP off there is one batch and it cannot: 3628 s, 2231 requests,
# zero faults, against four faulting PP2 arms.
#
# THE CHANGE. Widen the guard to both ends, in Python, at the point the indices are
# built. Out-of-range columns are clamped for the gather and then masked to
# NULL_BLOCK_ID, which is what the existing code already does for negative columns --
# the kernel's `state_idx != NULL_STATE_IDX` test then disables that program's stores.
# When every column is in range, which is every non-PP case, the result is unchanged.
#
# It logs the offending values when it fires, so this is a measurement as much as a
# fix. Three earlier theories died because nothing in them was instrumented; this one
# will say plainly whether it was right.
# =============================================================================
set -euo pipefail

echo "=== ckptcol-bounds: bound checkpoint_cols by the block table width ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda_metadata.py",
)
src = open(target).read()

if "[ckptcol]" in src:
    print("[ckptcol] already applied: " + target)
    sys.exit(0)

# The module has no logger of its own -- checked, not assumed -- and the warning below
# would NameError without one. Add it next to the imports it already has.
if "init_logger" not in src:
    IMPORT_ANCHOR = "from vllm.config import VllmConfig\n"
    if src.count(IMPORT_ANCHOR) != 1:
        sys.exit(
            "[ckptcol] FATAL: expected one VllmConfig import to anchor the logger to, "
            "found %d" % src.count(IMPORT_ANCHOR)
        )
    src = src.replace(
        IMPORT_ANCHOR,
        IMPORT_ANCHOR + "from vllm.logger import init_logger\n",
        1,
    )
    # After the import block, before the first definition.
    DEF_ANCHOR = "\n@dataclass\n"
    if DEF_ANCHOR not in src:
        sys.exit("[ckptcol] FATAL: no dataclass to place the logger before")
    src = src.replace(
        DEF_ANCHOR, "\nlogger = init_logger(__name__)\n" + DEF_ANCHOR, 1
    )

ANCHOR = """                checkpoint_state_indices = m.block_table_tensor[
                    request_rows_tensor, checkpoint_cols_tensor
                ]
                checkpoint_state_indices = torch.where(
                    checkpoint_cols_tensor >= 0,
                    checkpoint_state_indices,
                    NULL_BLOCK_ID,
                )
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[ckptcol] FATAL: expected one checkpoint_state_indices gather, found %d; the "
        "code moved and this needs re-deriving" % src.count(ANCHOR)
    )

ADDITION = """                # [ckptcol] The original guard checked only the lower bound.
                # checkpoint_cols is seq_len // block_size - 1 with seq_len taken
                # from seq_lens_cpu_upper_bound, so it can exceed the block table's
                # width; the gather below is a torch advanced index and reads out of
                # bounds. Either IndexKernel catches it, or it returns a plausible
                # wrong block id that _store_cache_checkpoints_kernel then uses to
                # address conv_state.
                _ckptcol_width = m.block_table_tensor.size(1)
                _ckptcol_ok = (checkpoint_cols_tensor >= 0) & (
                    checkpoint_cols_tensor < _ckptcol_width
                )
                if not bool(_ckptcol_ok.all()):
                    _ckptcol_bad = checkpoint_cols_tensor[~_ckptcol_ok]
                    logger.warning_once(
                        "[ckptcol] %d of %d checkpoint columns out of range for a "
                        "block table of width %d; max offending column %d",
                        int((~_ckptcol_ok).sum()),
                        int(checkpoint_cols_tensor.numel()),
                        _ckptcol_width,
                        int(_ckptcol_bad.max()),
                    )
                checkpoint_state_indices = m.block_table_tensor[
                    request_rows_tensor,
                    torch.where(
                        _ckptcol_ok,
                        checkpoint_cols_tensor,
                        torch.zeros_like(checkpoint_cols_tensor),
                    ),
                ]
                checkpoint_state_indices = torch.where(
                    _ckptcol_ok,
                    checkpoint_state_indices,
                    NULL_BLOCK_ID,
                )
"""

patched = src.replace(ANCHOR, ADDITION, 1)
compile(patched, target, "exec")
open(target, "w").write(patched)
print("[ckptcol] applied: " + target)
PY

# Re-read and import rather than trusting the write. A skipped edit here is
# indistinguishable from "the fix did not help" once the run is over.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda_metadata.py")).read()
if src.count("[ckptcol]") < 2:
    sys.exit("[ckptcol] FATAL: marker missing after write")
if "_ckptcol_ok," not in src:
    sys.exit("[ckptcol] FATAL: the NULL_BLOCK_ID mask still uses the old condition")

import vllm.models.kimi_k3.nvidia.kda_metadata as km

if "logger" not in dir(km):
    sys.exit("[ckptcol] FATAL: module has no logger; warning_once would NameError")
if "torch" not in dir(km):
    sys.exit("[ckptcol] FATAL: module has no torch in scope")
print("[ckptcol] verified: module imports, both bounds guarded, logger present")
PY

echo "=== ckptcol-bounds: done ==="
