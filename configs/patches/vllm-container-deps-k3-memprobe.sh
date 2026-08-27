#!/usr/bin/env bash
# Ask the tensors about their storage instead of inferring it from strides.
# =============================================================================
# WHY THIS EXISTS. The last four theories were derived by reading strides and base
# pointers and reasoning about what the memory underneath must look like. All four were
# wrong, and the most recent one was wrong for an embarrassing reason: the "37x overrun"
# was computed by subtracting pointers that belonged to eight different processes. Redone
# per process, the largest gap between conv_state bases is 4.64 GiB, exactly the view's
# own span, and two of the eight workers show no overlap at all.
#
# conv_state has stride(0)=442368 against a contiguous row of 13824, so it is either a
# slice of a much larger cache or a view that reaches past its data. Pointer arithmetic
# cannot tell those apart. The tensor can:
#
#     span   = (sum((size-1) * stride) + 1) * element_size
#     offset = storage_offset() * element_size
#     offset + span > untyped_storage().nbytes()   ->  the view reaches past its storage
#
# That is a fact, not an inference, and it ends this line of investigation either way.
#
# THE SECOND CANDIDATE, measured by the same probe. The fault is PP-only, and PP2 is what
# puts more than one microbatch in flight. checkpoint_state comes from
# current_workspace_manager().get_simultaneous(), a shared workspace, and its leading
# dimension was already seen changing between (1,12,128,128) and (2,12,128,128). The V1
# runner calls lock_workspace() after cudagraph capture (gpu_model_runner.py:7029); the
# V2 runner, which DSpark forces, never does. An earlier probe was read as clearing this,
# but it only watched for reallocation -- which happens once at startup -- and cannot see
# a buffer being reused across batches after capture, which is the thing that matters
# when captured graphs hold their arguments by weak reference.
#
# HOW THIS ONE AVOIDS THE TRAP THAT SPOILED THE LAST TWO PROBES. Both predecessors capped
# reports per worker (40, then 60), every worker hit the cap within the first couple of
# hundred launches, and the logs then went silent for the twenty minutes leading up to the
# fault. Silence read as "stable" when it meant "stopped looking".
#
# Nothing here is capped by call count. Reports are deduplicated by *value*: a storage
# violation is reported once per tensor, and a workspace identity once per distinct
# (pointer, shape, storage) triple. Both sets are small and bounded by the model, so the
# probe stays live to the end of the run at no cost.
#
# And it always emits one full baseline on the very first launch, with every number for
# all four tensors, whether or not anything is wrong. A silent log from this probe means
# the baseline was printed and nothing changed -- never that the question went unasked.
#
# Reads only. Fixes nothing.
# =============================================================================
set -euo pipefail

echo "=== memprobe: storage bounds and workspace identity at the checkpoint kernel ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda.py",
)
src = open(target).read()

if "[memprobe]" in src:
    print("[memprobe] already applied: " + target)
    sys.exit(0)

ANCHOR = """                        block_size = 256
                        _store_cache_checkpoints_kernel[
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[memprobe] FATAL: expected one checkpoint launch, found %d; another patch may "
        "have rewritten the grid" % src.count(ANCHOR)
    )

ADDITION = '''                        block_size = 256
                        # [memprobe] Diagnostic only. Deduplicated by value, never by
                        # call count -- the two previous probes went silent at their
                        # per-worker cap long before the fault and were misread as
                        # evidence of stability.
                        try:
                            _m = _MEMPROBE
                            _m["calls"] += 1
                            _tens = (
                                ("qkv", mixed_qkv_ns),
                                ("conv", conv_state),
                                ("ckpt", checkpoint_state),
                                ("recur", recurrent_state),
                            )

                            def _reach(t):
                                es = t.element_size()
                                span = (
                                    sum(
                                        (s - 1) * st
                                        for s, st in zip(t.shape, t.stride())
                                    )
                                    + 1
                                ) * es
                                off = t.storage_offset() * es
                                return off, span, t.untyped_storage().nbytes()

                            if not _m["baseline"]:
                                _m["baseline"] = True
                                for _nm, _t in _tens:
                                    _o, _sp, _nb = _reach(_t)
                                    logger.warning(
                                        "[memprobe] baseline %-5s shape=%s stride=%s "
                                        "offset=%d span=%d storage=%d fits=%s",
                                        _nm,
                                        tuple(_t.shape),
                                        tuple(_t.stride()),
                                        _o,
                                        _sp,
                                        _nb,
                                        _o + _sp <= _nb,
                                    )

                            for _nm, _t in _tens:
                                _o, _sp, _nb = _reach(_t)
                                if _o + _sp > _nb and _nm not in _m["bad"]:
                                    _m["bad"].add(_nm)
                                    logger.warning(
                                        "[memprobe] OVERRUN %s at call=%d: view reaches "
                                        "%d bytes but storage is %d (over by %d) | "
                                        "shape=%s stride=%s offset=%d",
                                        _nm,
                                        _m["calls"],
                                        _o + _sp,
                                        _nb,
                                        _o + _sp - _nb,
                                        tuple(_t.shape),
                                        tuple(_t.stride()),
                                        _o,
                                    )

                            _wk = (
                                checkpoint_state.data_ptr(),
                                tuple(checkpoint_state.shape),
                                checkpoint_state.untyped_storage().nbytes(),
                                checkpoint_state.storage_offset(),
                            )
                            if _wk not in _m["ws"]:
                                _m["ws"].add(_wk)
                                logger.warning(
                                    "[memprobe] workspace #%d at call=%d ptr=%s "
                                    "shape=%s storage=%d offset=%d",
                                    len(_m["ws"]),
                                    _m["calls"],
                                    hex(_wk[0]),
                                    _wk[1],
                                    _wk[2],
                                    _wk[3],
                                )
                        except Exception as _m_e:
                            if not _MEMPROBE["err"]:
                                _MEMPROBE["err"] = True
                                logger.warning("[memprobe] probe failed: %r", _m_e)
                        _store_cache_checkpoints_kernel[
'''

src = src.replace(ANCHOR, ADDITION, 1)

STATE_ANCHOR = "logger = init_logger(__name__)\n"
if src.count(STATE_ANCHOR) != 1:
    sys.exit(
        "[memprobe] FATAL: expected one module logger, found %d" % src.count(STATE_ANCHOR)
    )
src = src.replace(
    STATE_ANCHOR,
    STATE_ANCHOR
    + "\n# [memprobe] value-deduplicated state; no call-count cap\n"
    + '_MEMPROBE = {"calls": 0, "baseline": False, "bad": set(), "ws": set(),'
    + ' "err": False}\n',
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[memprobe] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
if src.count("[memprobe]") < 5:
    sys.exit("[memprobe] FATAL: markers missing after write")
if "_store_cache_checkpoints_kernel[" not in src:
    sys.exit("[memprobe] FATAL: the launch is gone")

import vllm.models.kimi_k3.nvidia.kda as kda

for name in ("logger", "_MEMPROBE"):
    if name not in dir(kda):
        sys.exit("[memprobe] FATAL: %s missing from the kda module" % name)
if kda._MEMPROBE["calls"] != 0 or kda._MEMPROBE["baseline"]:
    sys.exit("[memprobe] FATAL: probe state is not fresh")
print("[memprobe] verified: module imports, state fresh")
PY

echo "=== memprobe: done ==="
