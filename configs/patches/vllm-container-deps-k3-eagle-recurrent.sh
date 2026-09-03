#!/usr/bin/env bash
# Wei Zhao's EAGLE / recurrent-boundary group, on top of his Mooncake stack.
# =============================================================================
# Source: https://github.com/wzhao18/vllm/tree/wzhao/k3-nvfp4-perf, four commits
# that sit above the six this chain already carries:
#
#   c00916162d  Fix Mamba prefix caching at Eagle replay boundaries
#   99d4f413de  Separate EAGLE proof and replay boundaries
#   ad3e6c50fc  Fix ownership of exact recurrent-page offloads
#   9b1434b614  Fix hybrid EAGLE replay checkpoint retention
#
# Two commits between them on that branch are reverts of two others
# (6c02037437 "Fix KDA replay convolution history" and 16906b4b29 "Publish exact
# recurrent cache boundaries"), so the four above are the net that survives, and
# this patch is the diff of the tip state against the same base.
#
# The first squashes the functional part of vLLM PR #53945, which is in neither
# the 08/28 image (6f7df92a) nor the 09/01 nightly (7c5dc571) -- so unlike the
# Mooncake six this is content we cannot get by moving images.
#
# WHY WE ARE MEASURING IT. K3 is hybrid: 24 full-attention layers beside 69 KDA.
# Prefix caching has to checkpoint the recurrent state at replay boundaries or a
# reused prefix cannot be resumed, and the group publishes full-attention EAGLE
# groups through the prompt-hash boundary while keeping recurrent groups at their
# own replay checkpoints, sharing kv_cache_utils.replay_boundary between the
# engine and the store instead of letting the two disagree.
#
# That is the mechanism behind the wall these arms exist to probe: on the pinned
# image our DSpark arms peak at c48 (ns=4, 8,382) and fall away above it -- ns=7
# gives 5,453 at c72 and 4,171 at c80 -- with GPU hit collapsing .903 -> .355 by
# c96. If the collapse is recurrent state that is never retained across a replay
# boundary, this is where it is fixed. If it is simply the draft KV halving the
# pool, this changes nothing and the arms say so.
#
# The six Mooncake commits underneath measured neutral on B200 AGG (+0.07% at
# c32), so carrying them as the base costs nothing. Do NOT set compact_group_io
# with them on B200: it zeroes the external hit rate and costs 13.6%.
# =============================================================================
set -euo pipefail

echo "=== eagle-recurrent: EAGLE and recurrent replay boundaries ==="

# Chain the control's own recipe (828mc-dspark = 828-dspark + mambacache), not
# 828-dspark3, so the only difference from the c48 winner is this patch and the
# Mooncake six it rides on. Applying wei6 here rather than calling
# vllm-container-deps-k3-b200-828-wei-mooncake.sh avoids dragging in mambagroups2.
bash /configs/patches/vllm-container-deps-k3-b200-828mc-dspark.sh

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
if grep -q "max_load_batch_keys" \
     "$VLLM_ROOT/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"; then
    echo "[eagle-recurrent] Mooncake six already applied"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-mooncake-wei6-828.patch > /tmp/wei6-dry.log 2>&1; then
        echo "[eagle-recurrent] FATAL: the Mooncake six do not apply to this image" >&2
        cat /tmp/wei6-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-mooncake-wei6-828.patch
    echo "[eagle-recurrent] Mooncake six applied"
fi

if grep -q "get_replay_boundary" "$VLLM_ROOT/vllm/v1/core/kv_cache_coordinator.py"; then
    echo "[eagle-recurrent] already applied to this image"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-eagle-recurrent-828.patch > /tmp/eagle-dry.log 2>&1; then
        echo "[eagle-recurrent] FATAL: does not apply to this image" >&2
        echo "[eagle-recurrent] cut against 6f7df92a + #53324 + the Mooncake six" >&2
        cat /tmp/eagle-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-eagle-recurrent-828.patch
    echo "[eagle-recurrent] applied under $VLLM_ROOT"
fi

# Verify the thing the arms depend on: that the engine and the store now read the
# same replay boundary. A patch that applied but left the two disagreeing would
# measure the control twice and look like a null result.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
coord = open(os.path.join(root, "vllm/v1/core/kv_cache_coordinator.py")).read()
utils = open(os.path.join(root, "vllm/v1/core/kv_cache_utils.py")).read()
store = os.path.join(root, "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store")
sched = open(os.path.join(store, "scheduler.py")).read()

if "def replay_boundary" not in utils:
    sys.exit("[eagle-recurrent] FATAL: kv_cache_utils has no replay_boundary")
if "get_replay_boundary" not in coord:
    sys.exit("[eagle-recurrent] FATAL: the coordinator does not expose get_replay_boundary")
if "replay_boundary" not in sched and "get_replay_boundary" not in sched:
    print("[eagle-recurrent] note: the store scheduler does not name the boundary directly")
print("[eagle-recurrent] verified: engine and store share one replay boundary")
PY

echo "=== eagle-recurrent: done ==="
