#!/usr/bin/env bash
# Upstream PR #53614: compute the prefill checkpoint from the hash block, not the
# mamba block.
# =============================================================================
# WHY THIS ONE. The kernel that faults, _store_cache_checkpoints_kernel, was added by
# #52789. Every argument it receives has now been measured correct at the moment it runs
# -- indices from inside the kernel itself, strides, storage bounds, workspace identity --
# so the fault is not a bad address. What has never been checked is whether the checkpoint
# is exported at a position that means anything, and #53614 is exactly that defect.
#
# The nightly still carries #52789's original formula:
#
#     offset = seq_len // block_size * block_size - (seq_len - query_len)
#
# block_size is the mamba block size, and prefix_match_unit does not appear at all. Our
# arms run prefix-match-unit 128, so the prefix cache can resume at a hash-block boundary
# that is not a mamba-block boundary, and the offset is then measured from a base the
# cache never used. #53614 replaces it with
#
#     hash_block_size     = prefix_match_unit or block_size
#     checkpoint_position = get_mamba_prefill_checkpoint_position(seq_len, hash_block_size, ...)
#     offset              = checkpoint_position - query_start
#
# and adds two validity conditions the old code has no equivalent of:
#
#     query_start % checkpoint_alignment == 0
#     query_start + checkpoint_alignment <= checkpoint_position
#
# checkpoint_offsets becomes token_idx in the faulting kernel and state_indices is gated
# by the same `valid`. An offset that is in range but wrong writes state into a block the
# block table does not expect -- corruption first, fault later, with every index reading
# correct at every launch throughout.
#
# WHAT WOULD STILL BE UNEXPLAINED. The fault is only ever on PP stage 0, three runs of
# three, and nothing in this patch is PP-aware. A clean run here is a result, not an
# explanation.
#
# TWO THINGS THE FIRST VERSION OF THIS SCRIPT GOT WRONG, both caught before they could be
# read as "the fix does not work":
#
#   It used `git apply`. There is no git in this image. Every other patch script here
#   uses patch(1), and set -euo pipefail stopped the arm in setup rather than letting it
#   run unpatched.
#
#   It carried the PR as a commit series (Accept: ...v3.patch), which lists the same file
#   four times; patch(1) applies them in one pass and reports "Reversed (or previously
#   applied) patch detected". The combined diff (...v3.diff) touches each of five files
#   once and applies at fuzz 0.
# =============================================================================
set -euo pipefail

echo "=== pr53614: hash-block prefill checkpoints ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
KDA="$VLLM_ROOT/vllm/models/kimi_k3/nvidia/kda_metadata.py"

if grep -q "get_mamba_prefill_checkpoint_position" "$KDA"; then
    echo "[pr53614] already present in this image"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-pr53614.patch > /tmp/pr53614-dry.log 2>&1; then
        echo "[pr53614] FATAL: does not apply to this image" >&2
        cat /tmp/pr53614-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-pr53614.patch
    echo "[pr53614] applied under $VLLM_ROOT"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
kda = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda_metadata.py")).read()

# The point of the patch is that the old formula is gone and the new one is in. A patch
# that "applied" without changing this is the failure mode worth catching.
if "offset = seq_len // block_size * block_size - (seq_len - query_len)" in kda:
    sys.exit("[pr53614] FATAL: the old mamba-block offset formula is still there")
for need in ("get_mamba_prefill_checkpoint_position", "checkpoint_alignment",
             "query_start % checkpoint_alignment == 0"):
    if need not in kda:
        sys.exit("[pr53614] FATAL: %s missing after apply" % need)

import vllm.models.kimi_k3.nvidia.kda_metadata  # noqa: F401
from vllm.v1.kv_cache_interface import (  # noqa: F401
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_enabled,
)
print("[pr53614] verified: old formula gone, new helper importable, module loads")
PY

echo "=== pr53614: done ==="
