#!/usr/bin/env bash
# The empty-cache script plus the fork's speculative-decoding-under-pipeline-parallelism
# work, applied as a real patch.
# =============================================================================
# WHY A PATCH AND NOT A FLAG. Upstream refuses this outright. vLLM main, today, at
# vllm/v1/worker/gpu/spec_decode/dspark/utils.py:
#
#     if get_pp_group().world_size != 1:
#         raise NotImplementedError("DSpark does not support pipeline parallelism.")
#
# and SpeculativeConfig gives the draft the target's pipeline_parallel_size, so the draft
# architecture is asked to implement SupportsPP -- which K3DSparkForCausalLM does not.
# That is the NotImplementedError every one of our seven earlier PP2 arms died on.
#
# The support exists only in our vLLM fork, as three commits: 2111011d33 (adopting
# vllm-project/vllm#50514), 540423f2c7 and 503820ebdd. Their content, one line each:
#
#   - the draft's ParallelConfig stops inheriting PP size; the drafter runs whole on the
#     last stage and is never itself pipelined
#   - aux hidden states are relayed between stages as IntermediateTensors, behind a
#     per-model opt-in that only Kimi-K3 sets
#   - the last stage aliases a PPMissingLayer for the target embedding, so the real table
#     is loaded from the target checkpoint instead
#   - the draft loader falls back off fastsafetensors under PP, because its collective
#     runs over group.WORLD and the stages that never build a drafter never join
#   - the draft embedding lookup accepts an HF repo id, not just a directory
#
# PRE-FLIGHT, RUN BEFORE THIS SHIPPED. The patch was dry-run against two upstream
# snapshots that bracket this image -- cb8104839c (#50199) and main at #52836 -- and all
# 24 hunks apply to both, at offsets but without conflict. The fork touches an eighth file
# that is deliberately NOT in the patch; see the conditional at the bottom.
#
# WHY THIS ARM MUST SHIP WITH AN ACCURACY RUN. The adopting commit says what it fixed that
# five earlier attempts had missed: "draft tokens did not reach the earlier stages, so the
# first rank embedded PLACEHOLDER_TOKEN_ID(-1). Silent corruption -- we would not have
# seen it even if it reached serving." Every perf arm here runs ignore_eos and would
# report a fast, wrong server as a fast one. A GSM8K arm goes out alongside; treat the
# throughput number as unconfirmed until it lands.
#
# The patch is applied strictly: a --dry-run first, and a check afterwards that the
# refusal is gone. A mismatch fails the job rather than running a half-patched engine.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh

echo "=== k3-dspark-pp: apply spec-decode-under-PP ==="

PATCH=/configs/patches/k3-dspark-pp.patch
[ -f "$PATCH" ] || PATCH=/configs/k3-dspark-pp.patch
if [ ! -f "$PATCH" ]; then
  echo "[dspark-pp] FATAL: patch not found at /configs/patches/ or /configs/" >&2
  exit 1
fi

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[dspark-pp] vllm root: $VLLM_ROOT"

if grep -q "The drafter is instantiated only on the last pipeline stage" "$VLLM_ROOT/vllm/config/speculative.py"; then
  echo "[dspark-pp] already applied"
else
  if ! command -v patch >/dev/null 2>&1; then
    echo "[dspark-pp] installing patch(1)"
    apt-get update -qq && apt-get install -y -qq patch
  fi

  if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward < "$PATCH" > /tmp/dspark-pp-dry.log 2>&1; then
    echo "[dspark-pp] FATAL: patch does not apply to this image" >&2
    cat /tmp/dspark-pp-dry.log >&2
    exit 1
  fi

  patch -p1 -d "$VLLM_ROOT" --forward < "$PATCH"
  echo "[dspark-pp] applied"
fi

python3 - <<'PY'
import importlib.util, os, sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))

u = open(os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")).read()
if "DSpark does not support pipeline parallelism" in u:
    sys.exit("[dspark-pp] FATAL: the upstream refusal is still present after patching")
s = open(os.path.join(root, "vllm/config/speculative.py")).read()
if "pipeline_parallel_size=1" not in s:
    sys.exit("[dspark-pp] FATAL: draft ParallelConfig still inherits the target PP size")
print("[dspark-pp] verified: refusal removed, draft PP size pinned to 1")

# The eighth file the fork touches is not in the patch. Its change casts idx_mapping to
# int64 for index_fill_, which only accepts int64, on a branch reached solely from a
# non-last PP rank. Upstream hit the same bug independently and fixed it better in #50327
# by replacing index_fill_ with _fill_num_accepted_kernel, which takes the int32 mapping
# directly. Which side of that PR this image sits on cannot be read off its tag, so the
# file decides.
f = os.path.join(root, "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
src = open(f).read()
if "_fill_num_accepted_kernel" in src:
    print("[dspark-pp] mamba scalar update already uses the kernel (>= #50327)")
elif "idx_mapping.long()" in src:
    print("[dspark-pp] mamba int64 cast already present")
else:
    anchor = (
        "self.num_accepted_tokens_gpu.index_fill_(\n"
        "                0, idx_mapping, max(num_sampled, 1)\n"
        "            )"
    )
    if src.count(anchor) != 1:
        sys.exit(
            "[dspark-pp] FATAL: mamba_hybrid.py uses neither the kernel nor the "
            "index_fill_ shape we know how to cast; a non-last PP rank would fail on "
            "int32 indices"
        )
    patched = src.replace(
        anchor,
        "self.num_accepted_tokens_gpu.index_fill_(\n"
        "                # int64 only; idx_mapping is int32 and this branch is reached\n"
        "                # solely from the non-last PP rank, whose mapping is dense.\n"
        "                0, idx_mapping.long(), max(num_sampled, 1)\n"
        "            )",
    )
    compile(patched, f, "exec")
    open(f, "w").write(patched)
    print("[dspark-pp] mamba int64 cast applied")
PY

echo "=== k3-dspark-pp: done ==="
