#!/usr/bin/env bash
# Index probe, take two: watermarks instead of a head sample, plus the two cross-tensor
# constants the first probe never measured.
# =============================================================================
# WHY A SECOND PROBE. The first one produced the conclusion "every index is inside its
# tensor", and eight theories were built on top of that. The conclusion does not hold up:
#
#   probe hits 320 on the faulting node = 8 workers x the 40-report cap
#
# Every worker hit its cap, so what was recorded is the *earliest* forty occurrences.
# The illegal access arrives around request 230. The samples we reasoned from are from
# the first seconds of serving, when block ids are still small -- the probe printed
# state_idx maxima of 22, 24, 26, 35, 37, 39 against a conv_state of 5628 rows. Nothing
# in that sample says anything about what state_idx looks like near the crash.
#
# state_idx is a block id read out of block_table_tensor, and it grows as the run
# allocates. The first probe simply stopped watching before it mattered.
#
# WHAT CHANGES
#
# 1. No head sample. It reports only when a violation is *worse than any seen before* --
#    a watermark per quantity. Steady-state costs nothing, the log stays short, and the
#    last line before the crash is the worst case rather than the fortieth-earliest.
#
# 2. Two comparisons the first probe never made, both places where the kernel takes a
#    size from one tensor and applies it to another:
#
#      RECURRENT_ROW_SIZE = checkpoint_state[0].numel()      -- a workspace buffer
#      tl.store(recurrent_state_ptr + state_idx * stride + cols, mask=cols < RECURRENT_ROW_SIZE)
#                                                            -- into the KV cache
#
#      WIDTH = mixed_qkv_ns.shape[-1]                        -- the qkv projection
#      tl.store(conv_state_ptr + ... + width_idx * state_stride_1, ...)
#      with width_idx < WIDTH                                -- into conv_state's dim 1
#
#    checkpoint_state comes from the workspace manager and recurrent_state from the KV
#    cache; mixed_qkv and conv_state are likewise unrelated allocations. Nothing forces
#    either pair to agree, and neither pair has ever been printed.
#
# 3. A monotonic call counter, so a line can be placed early or late in the run.
#
# It fixes nothing. Eight fixes have been written against this fault from code reading
# and all eight failed; both facts that survived came from measurement, and one of those
# turns out to have been measured over the wrong window.
# =============================================================================
set -euo pipefail

echo "=== kdaprobe2: watermark the checkpoint kernel's indices and sizes ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda.py",
)
src = open(target).read()

if "[kdaprobe2]" in src:
    print("[kdaprobe2] already applied: " + target)
    sys.exit(0)

ANCHOR = """                        block_size = 256
                        _store_cache_checkpoints_kernel[
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[kdaprobe2] FATAL: expected one checkpoint launch, found %d; another patch "
        "may already have rewritten the grid" % src.count(ANCHOR)
    )

ADDITION = '''                        block_size = 256
                        # [kdaprobe2] Diagnostic only. Watermarks, not a head sample:
                        # the previous probe capped at 40 per worker and therefore only
                        # ever saw the first seconds of serving, while the fault arrives
                        # around request 230.
                        try:
                            _q = _KDAPROBE2
                            _q["calls"] += 1
                            _n = checkpoint_offsets.numel()
                            _sidx = checkpoint.state_indices.reshape(-1)[:_n]
                            _off = checkpoint_offsets.reshape(-1)[:_n].to(torch.int64)
                            _qsl = non_spec_query_start_loc[:_n].to(torch.int64)
                            _s_hi = int(_sidx.max().item())
                            _tok_hi = int((_qsl + _off).max().item()) - 1
                            _rrs = int(recurrent_row_size)
                            _rs_row = int(recurrent_state[0].numel())
                            _cs_dim1 = (
                                int(conv_state.shape[1])
                                if conv_state.dim() > 2
                                else -1
                            )
                            _w = int(width)
                            _worse = (
                                _s_hi > _q["s_hi"]
                                or _tok_hi > _q["tok_hi"]
                                or _rrs > _q["rrs"]
                            )
                            _cross = (_rrs > _rs_row) or (
                                _cs_dim1 >= 0 and _w > _cs_dim1
                            )
                            if _worse or (_cross and not _q["cross"]):
                                _q["s_hi"] = max(_q["s_hi"], _s_hi)
                                _q["tok_hi"] = max(_q["tok_hi"], _tok_hi)
                                _q["rrs"] = max(_q["rrs"], _rrs)
                                _q["cross"] = _q["cross"] or _cross
                                logger.warning(
                                    "[kdaprobe2] call=%d seqs=%d | state_idx_max=%d "
                                    "vs conv_rows=%d recur_rows=%d | token_idx_max=%d "
                                    "vs x_rows=%d | RECURRENT_ROW_SIZE=%d vs "
                                    "recurrent_state_row=%d | WIDTH=%d vs "
                                    "conv_state_dim1=%d | CROSS_OVERFLOW=%s",
                                    _q["calls"],
                                    _n,
                                    _s_hi,
                                    int(conv_state.shape[0]),
                                    int(recurrent_state.shape[0]),
                                    _tok_hi,
                                    int(mixed_qkv_ns.shape[0]),
                                    _rrs,
                                    _rs_row,
                                    _w,
                                    _cs_dim1,
                                    _cross,
                                )
                        except Exception as _q_e:
                            if not _KDAPROBE2["err"]:
                                _KDAPROBE2["err"] = True
                                logger.warning("[kdaprobe2] probe failed: %r", _q_e)
                        _store_cache_checkpoints_kernel[
'''

src = src.replace(ANCHOR, ADDITION, 1)

STATE_ANCHOR = "logger = init_logger(__name__)\n"
if src.count(STATE_ANCHOR) != 1:
    sys.exit(
        "[kdaprobe2] FATAL: expected one module logger, found %d"
        % src.count(STATE_ANCHOR)
    )
src = src.replace(
    STATE_ANCHOR,
    STATE_ANCHOR
    + '\n# [kdaprobe2] watermarks, shared across forwards\n'
    + '_KDAPROBE2 = {"calls": 0, "s_hi": -1, "tok_hi": -1, "rrs": -1,'
    + ' "cross": False, "err": False}\n',
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[kdaprobe2] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
if src.count("[kdaprobe2]") < 4:
    sys.exit("[kdaprobe2] FATAL: markers missing after write")
if "_store_cache_checkpoints_kernel[" not in src:
    sys.exit("[kdaprobe2] FATAL: the launch is gone")

import vllm.models.kimi_k3.nvidia.kda as kda

for name in ("logger", "torch", "_KDAPROBE2"):
    if name not in dir(kda):
        sys.exit("[kdaprobe2] FATAL: %s missing from the kda module" % name)
if kda._KDAPROBE2["calls"] != 0:
    sys.exit("[kdaprobe2] FATAL: watermark state is not fresh")
print("[kdaprobe2] verified: module imports, watermarks initialised")
PY

echo "=== kdaprobe2: done ==="
