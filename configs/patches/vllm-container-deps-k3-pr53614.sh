#!/usr/bin/env bash
# Upstream PR #53614: compute the prefill checkpoint from the hash block, not the
# mamba block.
# =============================================================================
# WHY THIS ONE. The kernel that faults, _store_cache_checkpoints_kernel, was added by
# #52789. Every argument it receives has now been measured correct at the moment it runs
# -- indices from inside the kernel itself, strides, storage bounds, workspace identity --
# so the fault is not a bad address. What has never been checked is whether the checkpoint
# is being exported at a position that makes sense, and #53614 is exactly that defect.
#
# The nightly still carries the original formula:
#
#     offset = seq_len // block_size * block_size - (seq_len - query_len)
#
# block_size here is the mamba block size, and prefix_match_unit does not appear at all.
# Our arms run prefix-match-unit 128, so the prefix cache can resume at a hash-block
# boundary that is not a mamba-block boundary; the offset is then measured from a base
# the cache never used. #53614 replaces it with
#
#     hash_block_size       = prefix_match_unit or block_size
#     checkpoint_position   = get_mamba_prefill_checkpoint_position(seq_len, hash_block_size, ...)
#     offset                = checkpoint_position - query_start
#
# and adds two validity conditions the old code has no equivalent of:
#
#     query_start % checkpoint_alignment == 0
#     query_start + checkpoint_alignment <= checkpoint_position
#
# checkpoint_offsets is what becomes token_idx in the faulting kernel, and state_indices
# is gated by the same `valid`. An offset that is in range but wrong writes state into a
# block the block table does not expect, which is corruption first and a fault later --
# consistent with every index reading correct at every launch.
#
# WHAT WOULD STILL BE UNEXPLAINED. The fault is only ever on PP stage 0, three runs out
# of three, and nothing in this patch is PP-aware. So a clean run here is a result, not
# an explanation; the asymmetry still needs one.
#
# Tests are stripped from the patch (they cannot run in the container and pull in
# fixtures the nightly does not have). Source files only: kda_metadata.py plus five
# v1/core files and kv_cache_interface.py, which is where the new helper lives.
# =============================================================================
set -euo pipefail

echo "=== pr53614: hash-block prefill checkpoints ==="

python3 - <<'PY'
import importlib.util
import os
import subprocess
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
patch = "/configs/patches/k3-pr53614.patch"

kda = os.path.join(root, "vllm/models/kimi_k3/nvidia/kda_metadata.py")
if "get_mamba_prefill_checkpoint_position" in open(kda).read():
    print("[pr53614] already present in this image")
    sys.exit(0)

chk = subprocess.run(["git", "apply", "--check", "-p1", patch], cwd=root,
                     capture_output=True, text=True)
if chk.returncode:
    sys.exit("[pr53614] FATAL: does not apply to this image:\n" + chk.stderr[:1200])
ap = subprocess.run(["git", "apply", "-p1", patch], cwd=root,
                    capture_output=True, text=True)
if ap.returncode:
    sys.exit("[pr53614] FATAL: apply failed after --check passed:\n" + ap.stderr[:1200])
print("[pr53614] applied under " + root)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
kda = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda_metadata.py")).read()

# The whole point of the patch is that the old formula is gone and the new one is in.
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
