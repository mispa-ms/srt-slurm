#!/usr/bin/env bash
# Watch the strides and base pointers the checkpoint kernel is compiled against.
# =============================================================================
# WHERE THIS SITS. Eight theories are dead and the index question is now closed
# properly: kdaprobe2 watermarked to call 82,297 and every quantity held --
#
#   state_idx_max=4859 vs conv_rows=5628   token_idx_max=7685 vs x_rows=8192
#   RECURRENT_ROW_SIZE=196608 = recurrent_state_row   WIDTH=4608 = conv_state_dim1
#
# So the values the kernel indexes with are right, and the extents it indexes into are
# right. What has never been measured is the third input to every one of those
# addresses: the strides.
#
# WHY STRIDES ARE A REAL SUSPECT HERE, not a fishing trip.
#
# 1. Eight of them are passed, and all eight are declared tl.constexpr:
#
#      x_stride_0/1, state_stride_0/1/2, checkpoint_stride_0,
#      recurrent_state_stride_0, checkpoint_offset_stride
#
#    Triton keys its compiled-kernel cache on constexpr values. A kernel compiled for
#    one set of strides is reused whenever the constexprs match, and the addresses are
#    baked in.
#
# 2. conv_state is not the tensor it looks like. kda.py:683 does
#
#      if not is_conv_state_dim_first():
#          conv_state = conv_state.transpose(-1, -2)
#
#    so the kernel may be handed a non-contiguous view whose strides are permuted. Shape
#    checks -- which is all we have done -- cannot see that. Two tensors with identical
#    shapes and different layouts index completely differently, and only the strides
#    distinguish them.
#
# 3. checkpoint_state comes from the workspace manager, recurrent_state and conv_state
#    from the KV cache, mixed_qkv from the projection. Four unrelated allocations whose
#    strides are captured together into one constexpr signature.
#
# WHAT THIS LOGS. On the first launch, and again whenever any stride or base pointer
# changes from what was last seen, it prints all eight strides, the four base pointers,
# the shapes, and whether each tensor is contiguous. Steady state is silent.
#
#   strides change mid-run      -> a cached kernel is being reused against a different
#                                  layout, which is the fault
#   base pointer changes        -> the tensor was reallocated under a captured graph
#   nothing ever changes        -> strides are not it either, and the remaining suspect
#                                  is the memory itself rather than any argument
#
# Fixes nothing. Three of the eight dead theories came from misreading our own
# instrumentation, so this one states its predicate plainly: it reports *changes*, and a
# silent log means the values were stable, not that they were never checked.
# =============================================================================
set -euo pipefail

echo "=== strideprobe: watch the checkpoint kernel's strides and base pointers ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda.py",
)
src = open(target).read()

if "[strideprobe]" in src:
    print("[strideprobe] already applied: " + target)
    sys.exit(0)

ANCHOR = """                        block_size = 256
                        _store_cache_checkpoints_kernel[
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[strideprobe] FATAL: expected one checkpoint launch, found %d; another patch "
        "may have rewritten the grid" % src.count(ANCHOR)
    )

ADDITION = '''                        block_size = 256
                        # [strideprobe] Diagnostic only. Every stride below is a
                        # tl.constexpr, so Triton caches the compiled kernel against
                        # them; conv_state may be a transposed view (kda.py:683). Report
                        # the first launch and any change thereafter.
                        try:
                            _sp = _STRIDEPROBE
                            _sp["calls"] += 1
                            _sig = (
                                mixed_qkv_ns.stride(0),
                                mixed_qkv_ns.stride(1),
                                conv_state.stride(0),
                                conv_state.stride(1),
                                conv_state.stride(2),
                                checkpoint_state.stride(0),
                                recurrent_state.stride(0),
                                checkpoint_offsets.stride(0),
                            )
                            _ptrs = (
                                mixed_qkv_ns.data_ptr(),
                                conv_state.data_ptr(),
                                checkpoint_state.data_ptr(),
                                recurrent_state.data_ptr(),
                            )
                            if _sig != _sp["sig"] or _ptrs != _sp["ptrs"]:
                                _what = []
                                if _sp["sig"] is not None and _sig != _sp["sig"]:
                                    _what.append("STRIDES_CHANGED")
                                if _sp["ptrs"] is not None and _ptrs != _sp["ptrs"]:
                                    _what.append("PTRS_CHANGED")
                                _sp["sig"] = _sig
                                _sp["ptrs"] = _ptrs
                                _sp["n"] += 1
                                if _sp["n"] <= 60:
                                    logger.warning(
                                        "[strideprobe] #%d call=%d %s | strides "
                                        "qkv=%s conv=%s ckpt=%s recur=%s off=%s | "
                                        "shapes qkv=%s conv=%s ckpt=%s recur=%s | "
                                        "contig qkv=%s conv=%s ckpt=%s recur=%s | "
                                        "ptrs=%s",
                                        _sp["n"],
                                        _sp["calls"],
                                        ",".join(_what) or "FIRST",
                                        _sig[0:2],
                                        _sig[2:5],
                                        _sig[5],
                                        _sig[6],
                                        _sig[7],
                                        tuple(mixed_qkv_ns.shape),
                                        tuple(conv_state.shape),
                                        tuple(checkpoint_state.shape),
                                        tuple(recurrent_state.shape),
                                        mixed_qkv_ns.is_contiguous(),
                                        conv_state.is_contiguous(),
                                        checkpoint_state.is_contiguous(),
                                        recurrent_state.is_contiguous(),
                                        tuple(hex(p) for p in _ptrs),
                                    )
                        except Exception as _sp_e:
                            if not _STRIDEPROBE["err"]:
                                _STRIDEPROBE["err"] = True
                                logger.warning("[strideprobe] probe failed: %r", _sp_e)
                        _store_cache_checkpoints_kernel[
'''

src = src.replace(ANCHOR, ADDITION, 1)

STATE_ANCHOR = "logger = init_logger(__name__)\n"
if src.count(STATE_ANCHOR) != 1:
    sys.exit(
        "[strideprobe] FATAL: expected one module logger, found %d"
        % src.count(STATE_ANCHOR)
    )
src = src.replace(
    STATE_ANCHOR,
    STATE_ANCHOR
    + '\n# [strideprobe] last-seen stride signature and base pointers\n'
    + '_STRIDEPROBE = {"calls": 0, "n": 0, "sig": None, "ptrs": None, "err": False}\n',
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[strideprobe] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
if src.count("[strideprobe]") < 4:
    sys.exit("[strideprobe] FATAL: markers missing after write")
if "_store_cache_checkpoints_kernel[" not in src:
    sys.exit("[strideprobe] FATAL: the launch is gone")

import vllm.models.kimi_k3.nvidia.kda as kda

for name in ("logger", "_STRIDEPROBE"):
    if name not in dir(kda):
        sys.exit("[strideprobe] FATAL: %s missing from the kda module" % name)
if kda._STRIDEPROBE["sig"] is not None:
    sys.exit("[strideprobe] FATAL: probe state is not fresh")
print("[strideprobe] verified: module imports, state fresh")
PY

echo "=== strideprobe: done ==="
