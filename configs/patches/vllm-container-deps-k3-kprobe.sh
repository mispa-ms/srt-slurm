#!/usr/bin/env bash
# Check the indices inside the kernel, where they are actually read.
# =============================================================================
# THE FLAW IN EVERY CHECK SO FAR. kdaprobe2, strideprobe and memprobe all read device
# tensors on the host, before an asynchronous launch, and concluded that the arguments
# were fine. But this kernel does not receive the indices as arguments -- it reads them
# itself, at execution time:
#
#     state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx)
#     checkpoint_offset = tl.load(checkpoint_offsets_ptr + seq_idx * ...)
#
# So "every index is in range to call 82,297" describes the values at *launch*. The
# kernel does not run at launch. If anything writes those buffers between the host read
# and the kernel's execution, all of that evidence is stale, and it would look exactly
# like what we have: every argument valid, every theory dead, the fault continuing.
#
# It also fits the one asymmetry we have. The fault is only ever on PP stage 0 -- 21/0
# and 17/0 across two runs, with both stages running the kernel a comparable number of
# times -- and stage 0 is the stage that runs ahead, so it is the one whose next
# microbatch's metadata is written while the previous one's kernels are still in flight.
#
# WHAT THE KERNEL DOES NOT CHECK. Reading it (kda.py:262-269):
#
#     valid_checkpoint = (state_idx != NULL_STATE_IDX) & (checkpoint_offset > 0)
#
# NULL_STATE_IDX is NULL_BLOCK_ID = 0. So state_idx is tested for being the sentinel and
# for nothing else -- there is no upper bound against conv_state's row count, and it then
# multiplies state_stride_0 = 442368 to form an address. token_idx is likewise unbounded:
# valid_conv masks on cols and on checkpoint_offset >= STATE_LEN, never on token_idx
# against the number of rows in x.
#
# WHAT THIS ARM DOES. Both at once, because they answer different halves:
#
#   Counts, on the device, with atomics -- so the numbers are the ones the kernel saw at
#   execution rather than the ones we saw at launch.
#   Guards, by adding the missing bounds to the two masks -- so if those indices are the
#   fault, the fault stops.
#
# Reading it, and each outcome points somewhere different:
#
#   IMA gone,   counters > 0   the indices are out of range at execution time and were in
#                              range at launch. Root cause, and the guard is the fix.
#   IMA gone,   counters == 0  the guard changed timing, not values. Do not claim a fix;
#                              the host read every 2000 calls is the likely difference.
#   IMA stays,  counters > 0   bad indices are real but are not the whole fault.
#   IMA stays,  counters == 0  the indices are genuinely fine when the kernel runs. The
#                              fault is the memory the address points at, not the index,
#                              and the weak references in the graph replay are next.
#
# The host reads the counters every 2000 calls -- roughly 42 times across a run, about
# one every 30 seconds. That is deliberate: each read synchronises, and a fast cadence
# would perturb the timing of the very thing being measured.
#
# The sentinel is 0 and is excluded before any guard is applied. That is stated because
# an earlier probe on this bug counted its own sentinel as a violation and three arms
# were built on the result.
# =============================================================================
set -euo pipefail

echo "=== kprobe: bound the indices inside the kernel, and count what the guard catches ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda.py",
)
src = open(target).read()

if "[kprobe]" in src:
    print("[kprobe] already applied: " + target)
    sys.exit(0)

# ── 1. kernel signature: two extents and the counter buffer ──────────────────
SIG = """    NULL_STATE_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
"""
if src.count(SIG) != 1:
    sys.exit("[kprobe] FATAL: expected one kernel signature tail, found %d" % src.count(SIG))
src = src.replace(
    SIG,
    """    NULL_STATE_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    kprobe_ptr,
    CONV_ROWS: tl.constexpr,
    X_ROWS: tl.constexpr,
):
""",
    1,
)

# ── 2. state_idx: count out-of-range, then exclude it from the mask ──────────
STATE = """    valid_checkpoint = (state_idx != NULL_STATE_IDX) & (checkpoint_offset > 0)
"""
if src.count(STATE) != 1:
    sys.exit("[kprobe] FATAL: expected one valid_checkpoint, found %d" % src.count(STATE))
src = src.replace(
    STATE,
    """    valid_checkpoint = (state_idx != NULL_STATE_IDX) & (checkpoint_offset > 0)
    # [kprobe] state_idx is read from device memory here, not passed in, so this is the
    # only place its value at execution time can be seen. The sentinel is already
    # excluded above; what is counted below is a genuine out-of-range row.
    _kp_si = state_idx.to(tl.int32)
    tl.atomic_max(kprobe_ptr + 0, _kp_si)
    tl.atomic_max(kprobe_ptr + 4, checkpoint_offset.to(tl.int32))
    _kp_bad_state = valid_checkpoint & ((_kp_si >= CONV_ROWS) | (_kp_si < 0))
    tl.atomic_add(kprobe_ptr + 1, tl.where(_kp_bad_state, 1, 0))
    tl.atomic_max(kprobe_ptr + 5, tl.where(_kp_bad_state, _kp_si, 0))
    valid_checkpoint = valid_checkpoint & (_kp_si >= 0) & (_kp_si < CONV_ROWS)
""",
    1,
)

# ── 3. token_idx: same, after it is computed and before it is used ───────────
TOK = """    token_idx = checkpoint_end - STATE_LEN + history_idx
"""
if src.count(TOK) != 1:
    sys.exit("[kprobe] FATAL: expected one token_idx, found %d" % src.count(TOK))
src = src.replace(
    TOK,
    """    token_idx = checkpoint_end - STATE_LEN + history_idx
    # [kprobe] token_idx indexes x_ptr and is bounded nowhere: valid_conv masks on cols
    # and on checkpoint_offset, never on the number of rows in x.
    _kp_ti = token_idx.to(tl.int32)
    tl.atomic_max(kprobe_ptr + 2, tl.max(tl.where(valid_conv, _kp_ti, 0)))
    _kp_bad_tok = valid_conv & ((_kp_ti < 0) | (_kp_ti >= X_ROWS))
    tl.atomic_add(kprobe_ptr + 3, tl.sum(_kp_bad_tok.to(tl.int32)))
    tl.atomic_max(kprobe_ptr + 6, tl.max(tl.where(_kp_bad_tok, _kp_ti, 0)))
    valid_conv = valid_conv & (_kp_ti >= 0) & (_kp_ti < X_ROWS)
""",
    1,
)

# ── 4. call site: allocate the buffer, pass the extents, read it rarely ──────
CALL = """                            NULL_BLOCK_ID,
                            block_size,
                        )
"""
if src.count(CALL) != 1:
    sys.exit("[kprobe] FATAL: expected one kernel call tail, found %d" % src.count(CALL))
src = src.replace(
    CALL,
    """                            NULL_BLOCK_ID,
                            block_size,
                            _kp_buf,
                            conv_state.shape[0],
                            mixed_qkv_ns.shape[0],
                        )
                        # [kprobe] Read every 2000 launches, ~42 times in a run. Each
                        # read synchronises, so a faster cadence would perturb the
                        # timing of the thing being measured.
                        _KPROBE["calls"] += 1
                        if _KPROBE["calls"] % 2000 == 0:
                            _v = _kp_buf.tolist()
                            if _v != _KPROBE["last"]:
                                _KPROBE["last"] = _v
                                logger.warning(
                                    "[kprobe] call=%d | state_max=%d state_oob=%d "
                                    "worst_state=%d vs CONV_ROWS=%d | token_max=%d "
                                    "token_oob=%d worst_token=%d vs X_ROWS=%d | "
                                    "offset_max=%d",
                                    _KPROBE["calls"], _v[0], _v[1], _v[5],
                                    conv_state.shape[0], _v[2], _v[3], _v[6],
                                    mixed_qkv_ns.shape[0], _v[4],
                                )
""",
    1,
)

# The buffer has to exist before the launch that uses it.
ANCHOR = """                        block_size = 256
"""
if src.count(ANCHOR) != 1:
    sys.exit("[kprobe] FATAL: expected one block_size, found %d" % src.count(ANCHOR))
src = src.replace(
    ANCHOR,
    """                        block_size = 256
                        # [kprobe] one int32 buffer per process, on the kernel's device
                        _kp_buf = _KPROBE["buf"]
                        if _kp_buf is None or _kp_buf.device != conv_state.device:
                            _kp_buf = torch.zeros(
                                8, dtype=torch.int32, device=conv_state.device
                            )
                            _KPROBE["buf"] = _kp_buf
""",
    1,
)

STATE_ANCHOR = "logger = init_logger(__name__)\n"
if src.count(STATE_ANCHOR) != 1:
    sys.exit("[kprobe] FATAL: expected one module logger, found %d" % src.count(STATE_ANCHOR))
src = src.replace(
    STATE_ANCHOR,
    STATE_ANCHOR
    + '\n# [kprobe] device-side counters, read on a slow cadence\n'
    + '_KPROBE = {"buf": None, "calls": 0, "last": None}\n',
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[kprobe] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import re
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")
src = open(path).read()

if src.count("[kprobe]") < 6:
    sys.exit("[kprobe] FATAL: markers missing after write (%d)" % src.count("[kprobe]"))

# The kernel signature and the call must agree, or Triton fails at the first launch --
# hours later, on the cluster, with a traceback that says nothing about this patch.
for name in ("kprobe_ptr", "CONV_ROWS: tl.constexpr", "X_ROWS: tl.constexpr"):
    if name not in src:
        sys.exit("[kprobe] FATAL: %s missing from the kernel signature" % name)
for arg in ("_kp_buf,", "conv_state.shape[0],", "mixed_qkv_ns.shape[0],"):
    if arg not in src:
        sys.exit("[kprobe] FATAL: %s missing from the call" % arg)
if "_kp_buf = _KPROBE" not in src:
    sys.exit("[kprobe] FATAL: buffer is never allocated")
if src.index("_kp_buf = _KPROBE") > src.index("                            _kp_buf,"):
    sys.exit("[kprobe] FATAL: buffer is allocated after it is passed")

# The guards must sit between the value and its use, not after it.
if src.index("valid_checkpoint = valid_checkpoint & (_kp_si >= 0)") > src.index("    valid_conv = ("):
    sys.exit("[kprobe] FATAL: the state guard lands after valid_conv is built")
if src.index("valid_conv = valid_conv & (_kp_ti >= 0)") > src.index("    values = tl.load("):
    sys.exit("[kprobe] FATAL: the token guard lands after the load it protects")

import vllm.models.kimi_k3.nvidia.kda as kda

for name in ("logger", "torch", "_KPROBE"):
    if name not in dir(kda):
        sys.exit("[kprobe] FATAL: %s missing from the kda module" % name)
if _ := kda._KPROBE["calls"]:
    sys.exit("[kprobe] FATAL: counter state is not fresh")
print("[kprobe] verified: signature and call agree, guards precede their uses")
PY

echo "=== kprobe: done ==="
