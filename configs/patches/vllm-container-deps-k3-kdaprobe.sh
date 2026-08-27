#!/usr/bin/env bash
# Log every index the KDA checkpoint kernel will use, against every bound it must obey.
# =============================================================================
# WHY A PROBE AND NOT A FIFTH FIX. Four theories about this illegal memory access have
# been written, patched and run, and all four were wrong:
#
#   shared metadata builder    fixed, still faulted
#   cp_gather_cache bounds     CUDA_LAUNCH_BLOCKING pointed at a different kernel
#   checkpoint grid too long   clamped, still faulted, and the mismatch logger it
#                              carried never fired -- the three lengths do agree
#   checkpoint column range    guarded, still faulted (40 in the full run)
#
# Each was plausible from reading the code. The pattern is the problem: read, guess,
# patch, burn an hour. The two facts that did survive both came from instrumentation,
# not from reading -- that the lengths agree, and that 7 of 8 checkpoint columns really
# do exceed the block table width. So this arm fixes nothing and measures everything.
#
# WHAT IS ACTUALLY ESTABLISHED. The launch, from blocking:
#
#   kimi_k3/nvidia/kda.py:904  _store_cache_checkpoints_kernel
#
# added by upstream PR #52789 on 2026-08-22 and absent from the pinned image whose PP2
# runs are clean. And the axis: PP off ran 3628 s and 2231 requests with zero faults,
# PP2 has faulted five times.
#
# WHAT THE KERNEL DEREFERENCES, and therefore what this logs:
#
#   state_idx      -> conv_state[state_idx]         and recurrent_state[state_idx]
#   token_idx      -> x[token_idx]   where token_idx = query_start_loc[seq]
#                                                    + checkpoint_offset
#                                                    - STATE_LEN + history_idx
#   seq_idx        -> the three per-request tensors
#
# Every one of those is computed here on the host, before the launch, and compared with
# the size of the tensor it will address. Only violations are printed, so a healthy
# forward costs three small reductions and no output.
#
# It is capped at 40 reports. The point is to see which bound breaks first and by how
# much, not to fill the log.
#
# CHAINED AFTER THE COLUMN GUARD ON PURPOSE. That guard is kept -- the out-of-range
# columns it reports are real and worth keeping out of the gather -- so anything this
# probe still finds is a second, separate problem rather than the one already known.
# =============================================================================
set -euo pipefail

echo "=== kdaprobe: log the checkpoint kernel's indices against their bounds ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda.py",
)
src = open(target).read()

if "[kdaprobe]" in src:
    print("[kdaprobe] already applied: " + target)
    sys.exit(0)

ANCHOR = """                        block_size = 256
                        _store_cache_checkpoints_kernel[
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[kdaprobe] FATAL: expected one checkpoint kernel launch, found %d"
        % src.count(ANCHOR)
    )

ADDITION = '''                        block_size = 256
                        # [kdaprobe] Diagnostic only -- changes nothing. Compare every
                        # index the kernel is about to use with the extent of the
                        # tensor it will address, and report only violations.
                        try:
                            _p_n = checkpoint_offsets.numel()
                            _p_qsl = non_spec_query_start_loc[:_p_n].to(torch.int64)
                            _p_off = checkpoint_offsets.reshape(-1)[:_p_n].to(
                                torch.int64
                            )
                            _p_sidx = checkpoint.state_indices.reshape(-1)[:_p_n].to(
                                torch.int64
                            )
                            _p_end = _p_qsl + _p_off
                            _p_tok_hi = int(_p_end.max().item()) - 1
                            _p_tok_lo = int((_p_end - state_len).min().item())
                            _p_s_hi = int(_p_sidx.max().item())
                            _p_s_lo = int(_p_sidx.min().item())
                            _p_bad = (
                                _p_tok_hi >= mixed_qkv_ns.shape[0]
                                or _p_tok_lo < 0
                                or _p_s_hi >= conv_state.shape[0]
                                or _p_s_hi >= recurrent_state.shape[0]
                            )
                            if _p_bad and _KDAPROBE["n"] < 40:
                                _KDAPROBE["n"] += 1
                                logger.warning(
                                    "[kdaprobe] #%d seqs=%d | token_idx [%d,%d] vs "
                                    "x_rows=%d | state_idx [%d,%d] vs conv=%d "
                                    "recur=%d | state_len=%d width=%d null=%d | "
                                    "off[min,max]=[%d,%d] qsl[min,max]=[%d,%d]",
                                    _KDAPROBE["n"],
                                    _p_n,
                                    _p_tok_lo,
                                    _p_tok_hi,
                                    mixed_qkv_ns.shape[0],
                                    _p_s_lo,
                                    _p_s_hi,
                                    conv_state.shape[0],
                                    recurrent_state.shape[0],
                                    state_len,
                                    width,
                                    NULL_BLOCK_ID,
                                    int(_p_off.min().item()),
                                    int(_p_off.max().item()),
                                    int(_p_qsl.min().item()),
                                    int(_p_qsl.max().item()),
                                )
                        except Exception as _p_e:  # never let the probe kill a run
                            if _KDAPROBE["n"] < 40:
                                _KDAPROBE["n"] += 1
                                logger.warning("[kdaprobe] probe failed: %r", _p_e)
                        _store_cache_checkpoints_kernel[
'''

src = src.replace(ANCHOR, ADDITION, 1)

# A module-level counter so the cap survives across forwards.
COUNTER_ANCHOR = "logger = init_logger(__name__)\n"
if src.count(COUNTER_ANCHOR) != 1:
    sys.exit(
        "[kdaprobe] FATAL: expected one module logger to anchor the counter to, found "
        "%d" % src.count(COUNTER_ANCHOR)
    )
src = src.replace(
    COUNTER_ANCHOR,
    COUNTER_ANCHOR + '\n# [kdaprobe] report cap, shared across forwards\n_KDAPROBE = {"n": 0}\n',
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[kdaprobe] applied: " + target)
PY

# Import it rather than trusting the write, and confirm the names the probe leans on
# actually exist in that module -- a NameError inside the try would be swallowed and
# the run would look clean while telling us nothing.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
if src.count("[kdaprobe]") < 3:
    sys.exit("[kdaprobe] FATAL: markers missing after write")

import vllm.models.kimi_k3.nvidia.kda as kda

for name in ("logger", "torch", "NULL_BLOCK_ID", "_KDAPROBE"):
    if name not in dir(kda):
        sys.exit("[kdaprobe] FATAL: %s is not in the kda module" % name)
print("[kdaprobe] verified: module imports, counter installed, names resolve")
PY

echo "=== kdaprobe: done ==="
