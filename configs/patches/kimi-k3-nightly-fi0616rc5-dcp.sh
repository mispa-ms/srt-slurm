#!/bin/bash
# kimi-k3-nightly-fi0616rc5.sh plus decode context parallelism for Kimi-K3.
#
# One delta against the script it wraps, so a -cur-tp8base- arm and a -cur-tp8dcp4-
# arm differ by exactly this patch plus decode-context-parallel-size.
#
# WHAT IT CARRIES
#
#   vllm-project/vllm#50484  [Kimi-K3] DCP support           (GirasoleY, 5dfad640d)
#   vllm-project/vllm#50493  DCP partial prefix cache hit    (GirasoleY, 9f409e3e2)
#
# Both are open and unmerged. Their merge-base is cb810483 -- the nightly this
# track runs -- so all seven commits cherry-pick onto the container's vLLM with
# no conflicts and the twelve shipped Python files apply with no fuzz. Verified
# with `patch -p1 --dry-run` against a pristine cb810483 tree before shipping.
#
# WHY BOTH. #50484 alone regresses prefix caching. Kimi-K3 is hybrid, so there
# are two KV groups; the attention group's block scales by DCP while the mamba
# group's does not, and base kv_cache_coordinator.py reads
#
#     enable_partial_hash_hits = dcp_world_size == 1 and any(...)
#
# so find_longest_cache_hit falls back from the hash block to the DCP-scaled
# group block -- 1536 -> 6144 tokens at DCP4. On a long-shared-prefix agentic
# workload that confounds the very thing the arm is trying to measure. #50493
# lifts that guard for the DCP case.
#
# WHAT IT DOES NOT CARRY. The PR also adds three direct symmetric-memory CUDA
# kernels (csrc/libtorch_stable/attention/dcp_utils/*.cu) and their CMakeLists
# entries. Building them means rebuilding the container, so they are left out;
# MLADCPManager falls back to the Triton/NCCL combine and gather. That fallback
# is NOT automatic -- VLLM_USE_DIRECT_DCP_{A2A,Q_GATHER,KV_GATHER} default to
# auto, which self-enables on B300 and then calls torch.ops._C.direct_dcp_*,
# which this image does not have. The arm's config must set all three to 0.
# This script checks for that at the end and fails loudly if they are unset.
#
# The chunked-context win the PR measures (TPOT p50 41.2 -> 31.0 ms) is not in
# the CUDA kernels -- it is all_gather_into_tensor writing straight into the
# persistent workspace instead of allocating and copying, so the fallback path
# gets it too.
#
# NOT COMPATIBLE WITH -mlaws.sh. wzhao18/vllm@2331dddd94 rewrites the same MLA
# chunked-prefill workspace sizing that #50484 replaces with
# align_mla_chunked_context_workspace_size. Stack this on the plain rc5 script
# only, and keep max-num-seqs pinned.
#
# KNOWN GAPS, the author's own words on 60b327588:
#
#   "Known gaps: plain-fp8-cache DCP all-gather and CUDA-graph e2e are
#    runtime-validation follow-ups"
#
# The plain-fp8 one held: the server came up and served for 15 minutes on B300
# at TP8/DCP4 with kv-cache-dtype fp8 (pipeline 61257869).
#
# The CUDA-graph one did not, and k3-dcp-cudagraph-pad-fix.patch below is ours.
# mask_dcp_empty_shards_ sizes repeat_interleave with output_size=lse.shape[0]
# while the repeats come from query_start_loc. A full CUDA graph replays at a
# captured batch size, so a 3-decode step runs on the size-4 graph and lse has
# a padding row query_start_loc does not account for. repeat_interleave then
# fires a device-side assert -- "output_size argument (4) must be the same as
# the sum of the elements in the repeats tensor (3)" -- and EngineCore dies.
# The fix carries the padding as one trailing masked segment, which is also the
# right semantics: those rows hold no query and must not reach the DCP combine.
# When there is no padding the emitted mask is identical to before.
#
# That masking is load-bearing. It is why the earlier revision of 60b327588
# could drop its "no full-CUDA-graph support under DCP" guard, so a broken mask
# means the guard is gone and nothing replaced it.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5.sh

echo "=== Kimi-K3 DCP patch (vllm#50484 @5dfad640d + #50493 @9f409e3e2) ==="
PATCH_FILE=/configs/patches/k3-dcp-pr50484-50493.patch
SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "DCP patch already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH_FILE"
fi

echo "=== DCP cudagraph padding fix (ours, on top of the PRs) ==="
FIX_FILE=/configs/patches/k3-dcp-cudagraph-pad-fix.patch
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$FIX_FILE" >/dev/null 2>&1; then
    echo "cudagraph padding fix already applied"
else
    patch -p1 --forward -d "$SITE" < "$FIX_FILE"
fi

python3 -c "
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent

present = [
    ('models/kimi_k3/nvidia/mla.py', 'self.dcp_manager: MLADCPManager | None = None'),
    ('v1/attention/ops/dcp_utils.py', 'class MLADCPManager'),
    ('v1/attention/ops/common.py', 'def mask_dcp_empty_shards_'),
    ('v1/attention/ops/dcp_alltoall.py', 'mask_dcp_empty_shards_(cp_attn_lse'),
    ('model_executor/layers/attention/mla_attention.py', 'align_mla_chunked_context_workspace_size'),
    ('envs.py', 'VLLM_USE_DIRECT_DCP_A2A'),
    ('v1/core/kv_cache_coordinator.py', 'dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size'),
    # ours, not upstream: without it a padded decode batch kills EngineCore
    ('v1/attention/ops/common.py', 'padding = lse.shape[0] - query_start_loc[-1]'),
]
for rel, marker in present:
    src = (root / rel).read_text()
    assert marker in src, f'DCP patch missing in {rel}: {marker}'

# The assert this whole arm exists to remove.
mla = (root / 'models/kimi_k3/nvidia/mla.py').read_text()
assert 'does not support context parallelism.' not in mla, (
    'DCP patch applied but the blanket context-parallel assert is still in mla.py'
)
print('DCP patch verified in', root)
"

# The direct symmetric-memory ops are not compiled into this image. Their env
# vars default to auto, which self-enables on Blackwell and then calls a kernel
# that does not exist. Refuse to hand back a container that would do that.
for v in VLLM_USE_DIRECT_DCP_A2A VLLM_USE_DIRECT_DCP_Q_GATHER VLLM_USE_DIRECT_DCP_KV_GATHER; do
    val="${!v-}"
    if [ "$val" != "0" ]; then
        echo "FATAL - $v is '${val:-<unset>}', must be 0. The direct DCP CUDA kernels"
        echo "are not built into this image; leaving these at their auto default makes"
        echo "MLADCPManager call torch.ops._C.direct_dcp_* and die at the first decode."
        exit 1
    fi
done
echo "direct DCP ops disabled (A2A/Q_GATHER/KV_GATHER = 0), using the Triton/NCCL fallback"
