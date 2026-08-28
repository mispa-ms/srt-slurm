#!/usr/bin/env bash
# Hanjie's fix: state_idx as int64, because the product overflows int32.
# =============================================================================
# THE BUG. conv_state.stride(0) is 442,368 (measured, strideprobe). state_idx is loaded
# as int32, so
#
#     state_idx * state_stride_0 > 2,147,483,647   at   state_idx > 4,854
#
# and kdaprobe2's watermark on the faulting node was 4,859 -- five past it. The product
# is 2,149,466,112, which wraps negative, and the store lands nowhere.
#
# WHY EVERY MEASUREMENT SAID THE ARGUMENTS WERE FINE, AND WAS RIGHT. The index is in
# range: 4,859 < conv_state.shape[0] = 5,628. The strides never changed. Every view fit
# its storage. The workspace never moved. All of that is true and none of it is the bug --
# the bug is the arithmetic that combines them, and nothing here was measuring products.
#
# The in-kernel probe missed it for a sharper reason: it read state_idx as
# `state_idx.to(tl.int32)` to feed an int32 atomic, so it inspected exactly the value that
# is fine and never the multiplication that is not.
#
# IT ALSO EXPLAINS THE ONE ASYMMETRY NOTHING ELSE DID. The fault is only ever on PP stage
# 0, three runs of three. Our own kprobe numbers: stage 0 reached state_idx 4,762 while
# stage 1 reached 4,417. Stage 0 runs against higher block ids, so stage 0 is the one that
# crosses 4,854. And the ~19.5-minute delay is how long allocation takes to get there.
#
# Only conv_state overflows: recurrent_state's stride is 221,184 (threshold 9,708),
# x_stride_0 is 6,288, and checkpoint_stride_0 multiplies a sequence index. One product
# of the four, and the fault is in the store that uses it.
#
# THE FIX is one cast, from Hanjie Qiu:
#
#     -    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx)
#     +    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx).to(tl.int64)
#
# Applied verbatim. No probe, no guard: this arm is a confirmation, and it runs the full
# 3600 s because a shorter one could only fail to disprove it.
# =============================================================================
set -euo pipefail

echo "=== int64idx: state_idx to int64 in the checkpoint kernel ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda.py",
)
src = open(target).read()

FIXED = "state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx).to(tl.int64)"
if FIXED in src:
    print("[int64idx] already present in this image")
    sys.exit(0)

ANCHOR = "    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx)\n"
if src.count(ANCHOR) != 1:
    sys.exit(
        "[int64idx] FATAL: expected one state_idx load, found %d -- has another patch "
        "already rewritten this kernel?" % src.count(ANCHOR)
    )

src = src.replace(ANCHOR, "    " + FIXED + "\n", 1)
compile(src, target, "exec")
open(target, "w").write(src)
print("[int64idx] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import re
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")
src = open(path).read()

if ".to(tl.int64)" not in src:
    sys.exit("[int64idx] FATAL: the cast is not in the file after writing it")
if re.search(r"state_idx = tl\.load\(checkpoint_state_indices_ptr \+ seq_idx\)\s*$",
             src, re.M):
    sys.exit("[int64idx] FATAL: an uncast state_idx load is still present")

# The cast is worth nothing if it does not reach the multiplication that overflows.
if "state_idx * state_stride_0" not in src:
    sys.exit("[int64idx] FATAL: the conv_state address no longer uses state_stride_0")

import vllm.models.kimi_k3.nvidia.kda  # noqa: F401

print("[int64idx] verified: cast present, no uncast load left, address still uses it")
PY

echo "=== int64idx: done ==="
