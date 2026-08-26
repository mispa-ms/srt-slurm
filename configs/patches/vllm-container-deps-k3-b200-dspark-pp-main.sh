#!/usr/bin/env bash
# DSpark under pipeline parallelism, for a CURRENT vLLM nightly.
# =============================================================================
# The sibling of vllm-container-deps-k3-b200-dspark-pp-xin.sh. That one carries the
# three commits as they sit on xinli-sw/vllm:k3-agent-all, which is the tree the
# 728d3ad image was built from. This one carries them rebased onto upstream main,
# for the nightly images that have moved past it.
#
# WHY A SECOND PATCH. main still refuses spec decode under PP -- the
# NotImplementedError is right there in spec_decode/dspark/utils.py -- but the file
# around it has moved, and the older patch fails 3 of its hunks. The three commits
# were cherry-picked onto main and two conflicts resolved by union rather than by
# choosing a side:
#
#   imports          main added _resolve_dspark_attention_backend for the DeepSeek-V4
#                    drafter and the AttentionBackendEnum/ModelConfig imports it needs;
#                    ours added get_model, PPMissingLayer and the eagle utils. Both
#                    kept -- they do different things and only the import block
#                    overlapped.
#   embed sharing    main added a vocab-size guard, ours added the PPMissingLayer
#                    resolution. Both conditions kept.
#
# VERIFIED BEFORE SHIPPING, unlike the last patch this workstream added blind:
#   - ten assertions on the merged tree (refusal gone, draft PP pinned, aux relay
#     opt-in and pack/unpack present, PP embed loader present, main's DSV4 helper
#     still there, mamba on the kernel and not our int64 cast) -- all pass
#   - six touched files compile
#   - dry-run against a9a17e70, the tree the current nightly image was built from:
#     0 failed, 0 fuzz
#
# The applier is strict (--fuzz=0) and the guard below skips when the image already
# carries the change, so this is safe on either an old or a future nightly.
# =============================================================================
set -euo pipefail


echo "=== k3-dspark-pp-main: apply spec-decode-under-PP, rebased onto upstream main ==="

PATCH=/configs/patches/k3-dspark-pp-main.patch
[ -f "$PATCH" ] || PATCH=/configs/k3-dspark-pp-main.patch
if [ ! -f "$PATCH" ]; then
  echo "[dspark-pp-main] FATAL: patch not found at /configs/patches/ or /configs/" >&2
  exit 1
fi

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[dspark-pp-main] vllm root: $VLLM_ROOT"

if grep -q "The drafter is instantiated only on the last pipeline stage" "$VLLM_ROOT/vllm/config/speculative.py"; then
  echo "[dspark-pp-main] already applied -- the image was built from a tree that has it"
else
  if ! command -v patch >/dev/null 2>&1; then
    echo "[dspark-pp-main] installing patch(1)"
    apt-get update -qq && apt-get install -y -qq patch
  fi

  # No fuzz allowed. This patch was derived from the branch and verified to apply to
  # this image exactly; if it suddenly needs to guess, the image moved and the run
  # should stop rather than quietly test something else.
  if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 < "$PATCH" > /tmp/dspark-pp-main-dry.log 2>&1; then
    echo "[dspark-pp-main] FATAL: patch does not apply cleanly to this image" >&2
    cat /tmp/dspark-pp-main-dry.log >&2
    exit 1
  fi

  patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < "$PATCH"
  echo "[dspark-pp-main] applied"
fi

python3 - <<'PY'
import importlib.util, os, sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))

u = open(os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")).read()
if "DSpark does not support pipeline parallelism" in u:
    sys.exit("[dspark-pp-main] FATAL: the upstream refusal is still present after patching")
s = open(os.path.join(root, "vllm/config/speculative.py")).read()
if "pipeline_parallel_size=1" not in s:
    sys.exit("[dspark-pp-main] FATAL: draft ParallelConfig still inherits the target PP size")

# The hand-resolved half. Their kernel must be what survived, not our index_fill_ cast:
# if the cast is present, the merge took the wrong side and this run would be measuring
# our workaround rather than the branch.
m = open(os.path.join(root, "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")).read()
if "idx_mapping.long()" in m:
    sys.exit("[dspark-pp-main] FATAL: the int64 cast is present; the mamba conflict "
             "resolved the wrong way and this is not the branch's code")
if "_fill_num_accepted_kernel" not in m:
    sys.exit("[dspark-pp-main] FATAL: neither the kernel nor the cast is present in "
             "mamba_hybrid.py; this image is not the one this patch was derived against")

# The other half: the aux-unpack branch has to sit in front of the tap collection, not
# somewhere the fuzzy match put it.
k = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/model.py")).read()
if "unpack_aux_hidden_states(intermediate_tensors)" not in k:
    sys.exit("[dspark-pp-main] FATAL: the PP aux-unpack branch is missing from model.py")
if "elif self.start_layer in self.aux_hidden_state_layers:" not in k:
    sys.exit("[dspark-pp-main] FATAL: model.py has the unpack branch but not the elif it "
             "must guard; the insert landed in the wrong place")
print("[dspark-pp-main] verified: refusal gone, draft PP pinned, mamba kept the kernel, "
      "aux-unpack sits ahead of the tap collection")
PY


# emptycache runs AFTER the patch, not before. Both edit
# vllm/v1/worker/gpu/model_runner.py, and emptycache inserts lines above the speculator
# load -- which shifts every hunk this patch expects further down. Applied first, it made
# the strict --fuzz=0 applier reject the whole patch:
#
#   [emptycache] applied: .../model_runner.py
#   [dspark-pp-main] FATAL: patch does not apply cleanly to this image
#
# The xin script chains it first and gets away with it only because the image it targets
# already carries the change, so its skip-guard fires and nothing is edited. In this
# order the patch lands on the tree it was derived against, and emptycache -- which
# matches its anchor as a whole line at any indent -- then finds it moved but intact.
# Verified in that order against a9a17e70: patch clean, emptycache clean.
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh

echo "=== k3-dspark-pp-main: done ==="
