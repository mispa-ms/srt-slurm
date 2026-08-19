#!/usr/bin/env bash
# The same spec-decode-under-PP change, but taken from Xin's branch instead of ours.
# =============================================================================
# WHY THIS EXISTS. Our three commits now sit on xinli-sw/vllm:k3-agent-all, the branch
# the image we run was built from, and an image will be built from that branch. Getting
# them there needed two conflicts resolved by hand:
#
#   mamba_hybrid.py  -- kept their side entirely. They already carry upstream #50327's
#                       _fill_num_accepted_kernel, which takes the int32 idx_mapping
#                       directly and skips -1 sentinels, so it supersedes our
#                       index_fill_(idx_mapping.long()) workaround and is safer.
#   kimi_k3/model.py -- positional only. Their SP all-gather restructuring moved the
#                       lines; the PP aux-unpack branch was placed at the real site.
#
# A hand-resolved merge is exactly the thing to confirm by measurement before someone
# builds an image from it. So this script applies the patch as it now exists ON THEIR
# BRANCH -- git diff 5ee73d93dc89..728d3ad09ffc -- rather than our original.
#
# WHAT WE ALREADY KNOW WITHOUT RUNNING IT. Excluding the new test file, the two patches
# add and remove the SAME 396 lines; every difference between them is context, from
# their tree having drifted (fused_qkv_a_proj, import block formatting). And this one
# applies to the running image with zero failures and zero fuzz, where ours needed
# fuzz 3. So this run is a confirmation, not a discovery -- but the arms it gates are
# the ones we would quote, so it is worth the hour.
#
# WHAT THIS DOES NOT CARRY. The empty_cache()-before-draft-load fix is not in those
# three commits and is not on their branch. It stays where it is, in the chained
# emptycache script, because DSpark + DCP8 + direct on DP2/EP16 dies without it.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh

echo "=== k3-dspark-pp-xin: apply spec-decode-under-PP, as it stands on k3-agent-all ==="

PATCH=/configs/patches/k3-dspark-pp-xin.patch
[ -f "$PATCH" ] || PATCH=/configs/k3-dspark-pp-xin.patch
if [ ! -f "$PATCH" ]; then
  echo "[dspark-pp-xin] FATAL: patch not found at /configs/patches/ or /configs/" >&2
  exit 1
fi

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[dspark-pp-xin] vllm root: $VLLM_ROOT"

if grep -q "The drafter is instantiated only on the last pipeline stage" "$VLLM_ROOT/vllm/config/speculative.py"; then
  echo "[dspark-pp-xin] already applied -- the image was built from a tree that has it"
else
  if ! command -v patch >/dev/null 2>&1; then
    echo "[dspark-pp-xin] installing patch(1)"
    apt-get update -qq && apt-get install -y -qq patch
  fi

  # No fuzz allowed. This patch was derived from the branch and verified to apply to
  # this image exactly; if it suddenly needs to guess, the image moved and the run
  # should stop rather than quietly test something else.
  if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 < "$PATCH" > /tmp/dspark-pp-xin-dry.log 2>&1; then
    echo "[dspark-pp-xin] FATAL: patch does not apply cleanly to this image" >&2
    cat /tmp/dspark-pp-xin-dry.log >&2
    exit 1
  fi

  patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < "$PATCH"
  echo "[dspark-pp-xin] applied"
fi

python3 - <<'PY'
import importlib.util, os, sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))

u = open(os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")).read()
if "DSpark does not support pipeline parallelism" in u:
    sys.exit("[dspark-pp-xin] FATAL: the upstream refusal is still present after patching")
s = open(os.path.join(root, "vllm/config/speculative.py")).read()
if "pipeline_parallel_size=1" not in s:
    sys.exit("[dspark-pp-xin] FATAL: draft ParallelConfig still inherits the target PP size")

# The hand-resolved half. Their kernel must be what survived, not our index_fill_ cast:
# if the cast is present, the merge took the wrong side and this run would be measuring
# our workaround rather than the branch.
m = open(os.path.join(root, "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")).read()
if "idx_mapping.long()" in m:
    sys.exit("[dspark-pp-xin] FATAL: the int64 cast is present; the mamba conflict "
             "resolved the wrong way and this is not the branch's code")
if "_fill_num_accepted_kernel" not in m:
    sys.exit("[dspark-pp-xin] FATAL: neither the kernel nor the cast is present in "
             "mamba_hybrid.py; this image is not the one this patch was derived against")

# The other half: the aux-unpack branch has to sit in front of the tap collection, not
# somewhere the fuzzy match put it.
k = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/model.py")).read()
if "unpack_aux_hidden_states(intermediate_tensors)" not in k:
    sys.exit("[dspark-pp-xin] FATAL: the PP aux-unpack branch is missing from model.py")
if "elif self.start_layer in self.aux_hidden_state_layers:" not in k:
    sys.exit("[dspark-pp-xin] FATAL: model.py has the unpack branch but not the elif it "
             "must guard; the insert landed in the wrong place")
print("[dspark-pp-xin] verified: refusal gone, draft PP pinned, mamba kept the kernel, "
      "aux-unpack sits ahead of the tap collection")
PY

echo "=== k3-dspark-pp-xin: done ==="
