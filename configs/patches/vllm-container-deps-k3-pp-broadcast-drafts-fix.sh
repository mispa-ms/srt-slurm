#!/usr/bin/env bash
# Fix the PP last-stage draft broadcast race (the third 09/08 wall).
# =============================================================================
# THE WALL. K3 TP8xPP2xDCP8 with a DSpark drafter on nightly 9ea8f3ff: PP1
# alone dies a few requests into serving with the ATen device-side assert
# `vectorized_gather_kernel: ind >= 0 && ind < ind_dim_size`, surfacing at
# whichever sync comes first (target MLA prefill, or the Mooncake worker
# thread). It reproduces on B200 AGG with Mooncake (66819512, 66843721) and on
# GB300 disagg with NIXL push (66840854), and it VANISHES under
# CUDA_LAUNCH_BLOCKING=1 on both machines (66853246, 66850730) -- a race.
#
# THE RACE. #50514 (d87a440f88) made the last rank broadcast the drafter's
# proposals to the other stages:
#
#     with torch.cuda.stream(self.broadcast_stream):
#         self.broadcast_stream.wait_stream(self.main_stream)
#         send = draft_tokens[input_batch.idx_mapping].contiguous()   # <-- gather on the side stream
#         torch.distributed.broadcast(send, ...)
#         send.record_stream(self.broadcast_stream)
#
# `input_batch.idx_mapping` is a per-step device tensor
# (`async_copy_to_gpu(idx_mapping_np)`, model_runner.py:1221) allocated on the
# main stream and dropped with the InputBatch when the step ends. Nothing
# records the side stream on it, so the caching allocator may hand its memory
# to the next step's main-stream allocations while the side-stream gather is
# still queued; the gather then reads whatever landed there as block ids ->
# out-of-bounds index. Our retired carry broadcast the whole `draft_tokens`
# buffer (no gather), which is why 08/28 never saw this.
#
# UPSTREAM. vllm#55745 (eastwood-c) fixes the same race with
# `input_batch.idx_mapping.record_stream(self.broadcast_stream)`; merged 2026-09-09
# (e8064a96d0). Images that carry it are detected below and left alone.
#
# THE FIX. Gather on the main stream, where `idx_mapping` is both valid and
# ordered, and let the side stream only own the broadcast. `send` is then a
# main-stream allocation used on the side stream, which the existing
# `send.record_stream(self.broadcast_stream)` covers.
# =============================================================================
set -euo pipefail

echo "=== k3-pp-broadcast-drafts-fix: gather drafts on the main stream ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
PU="$VLLM_ROOT/vllm/v1/worker/gpu/pp_utils.py"
[ -f "$PU" ] || { echo "[bcfix] FATAL: no pp_utils.py in this image" >&2; exit 1; }

python3 - "$PU" <<'PY'
import ast, sys
path = sys.argv[1]
src = open(path).read()
old = (
    "        with torch.cuda.stream(self.broadcast_stream):\n"
    "            self.broadcast_stream.wait_stream(self.main_stream)\n"
    "            send = draft_tokens[input_batch.idx_mapping].contiguous()\n"
    "            torch.distributed.broadcast(\n"
    "                send, src=self.last_rank, group=self.broadcast_group\n"
    "            )\n"
    "            send.record_stream(self.broadcast_stream)\n"
)
new = (
    "        # Gather on the main stream: `idx_mapping` is a per-step tensor that the\n"
    "        # caching allocator may reuse as soon as the host moves on, and a\n"
    "        # side-stream read of it races that reuse (device-side index assert on\n"
    "        # the last PP stage). `send` is then a main-stream allocation used on the\n"
    "        # broadcast stream, which record_stream below covers.\n"
    "        send = draft_tokens[input_batch.idx_mapping].contiguous()\n"
    "        with torch.cuda.stream(self.broadcast_stream):\n"
    "            self.broadcast_stream.wait_stream(self.main_stream)\n"
    "            torch.distributed.broadcast(\n"
    "                send, src=self.last_rank, group=self.broadcast_group\n"
    "            )\n"
    "            send.record_stream(self.broadcast_stream)\n"
)
if new in src:
    print("[bcfix] already applied"); sys.exit(0)
# Upstream's own fix (vllm#55745, merged 2026-09-09 as e8064a96d0) keeps the gather on the
# side stream and records that stream on idx_mapping instead. Equivalent; nothing to do.
if "input_batch.idx_mapping.record_stream(self.broadcast_stream)" in src:
    print("[bcfix] upstream #55745 present in this image (record_stream on idx_mapping); skipping"); sys.exit(0)
if src.count(old) != 1:
    sys.exit(f"[bcfix] FATAL: expected exactly one broadcast_drafts block, found {src.count(old)}; image differs from 9ea8f3ff")
# The block must be inside broadcast_drafts, not receive().
head = src[: src.index(old)]
if head.rfind("def broadcast_drafts") < head.rfind("def receive"):
    sys.exit("[bcfix] FATAL: the matched block is not in broadcast_drafts")
out = src.replace(old, new)
ast.parse(out)
open(path, "w").write(out)
print("[bcfix] applied: draft gather moved to the main stream in broadcast_drafts")
PY

echo "=== k3-pp-broadcast-drafts-fix: done ==="
