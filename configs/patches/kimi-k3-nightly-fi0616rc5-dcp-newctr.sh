#!/bin/bash
# Decode context parallelism for Kimi-K3, rebased onto the 2026-08-07 nightly.
#
# This is kimi-k3-nightly-fi0616rc5-dcp.sh moved forward one container. Keep both:
# the cb810483 script is what the whole measured DCP ladder ran on, and nothing
# should be re-attributed to it once the stack underneath changes.
#
# WHAT MOVED
#
#   container   cb810483 (2026-07-27)  ->  c810e5ee (2026-08-07 06:11)
#   #50484      5dfad640d              ->  b36b196c1
#   #50493      9f409e3e2c                 (unchanged, still open)
#
# c810e5ee is 153 commits past cb810483. Two of them matter here.
#
#   #50613  [Attention][MLA] Per-request scheduling for MLA chunked context
#           Rewrites the chunked-context path so prefill chunks are fit to the
#           available workspace instead of forced to a common size. That is the
#           exact path _context_parallel_compute_prefill_context runs through,
#           and it is the reason this arm exists: three DCP8 runs and one DCP4
#           run died on IndexKernel.cu:111 "index out of bounds" somewhere in
#           there, and it is still unlocated. Landed 2026-08-06, validated
#           upstream with DCP=2 plus prefix caching.
#
#   #50404  [Model] Fix Kimi-K3 MLA with disabled context parallelism
#           Also touches models/kimi_k3/nvidia/mla.py.
#
# WHY THE PATCH IS SMALLER: 12 files -> 10
#
# Our cb810483 patch carried block_pool.py and kv_cache_manager.py, from
# 3120e9980f and bcbb9e26bb on the k3-dcp-kv-cache branch. Both have since
# merged upstream, so against c810e5ee they are net-zero and drop out. What is
# left of #50493 is the eleven-line guard in kv_cache_coordinator.py, which is
# the load-bearing part -- base vLLM still reads
#
#     enable_partial_hash_hits = dcp_world_size == 1 and any(...)
#
# so without it prefix caching silently falls back from the 1536-token hash
# block to the DCP-scaled group block, and prefix-match-unit is ignored.
#
# OUR CUDA-GRAPH FIX IS GONE, ON PURPOSE
#
# k3-dcp-cudagraph-pad-fix.patch is NOT applied here and must not be. The author
# fixed the same defect independently in 76b2e9d45e ("Fix DCP empty-shard
# masking for padded graphs", 2026-08-06), and the fix is ours in method as well
# as effect -- searchsorted over query_start_loc instead of expanding repeats.
# The two differ only in how the padding tail is spelt: theirs clamps the index
# and ORs in `row >= query_start_loc[-1]`, ours appends a sentinel flag so an
# out-of-range index masks by construction. Applying ours on top fails cleanly
# (`Hunk #1 FAILED at 22`), which is the check that caught the overlap.
#
# OUR OFFLOAD FIX IS STILL REQUIRED
#
# k3-dcp-offload-hybrid-fix.patch still applies with no fuzz, and nothing
# upstream replaced it: simple_kv_offload/manager.py at c810e5ee still scales
# every group's block by cp_world_size at four sites and still asserts
# num_external_tokens is scheduler-block aligned.
#
# UNCHANGED FROM THE OLD SCRIPT
#
# The direct symmetric-memory CUDA kernels are still not built into the image,
# so VLLM_USE_DIRECT_DCP_{A2A,Q_GATHER,KV_GATHER} must all be 0. Checked below.

set -euo pipefail

DCP_BASE_SCRIPT="${DCP_BASE_SCRIPT:-/configs/patches/kimi-k3-nightly-fi0616rc5.sh}"
echo "=== DCP base stack: ${DCP_BASE_SCRIPT} ==="
bash "$DCP_BASE_SCRIPT"

echo "=== Kimi-K3 DCP patch, new container (vllm#50484 @b36b196c1 + #50493 @9f409e3e2c) ==="
PATCH_FILE=/configs/patches/k3-dcp-pr50484-50493-newctr.patch
SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "DCP patch already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH_FILE"
fi

echo "=== k3-dcp-offload-hybrid-fix (ours, on top of the PRs) ==="
FIX_FILE=/configs/patches/k3-dcp-offload-hybrid-fix.patch
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$FIX_FILE" >/dev/null 2>&1; then
    echo "offload fix already applied"
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
    # the author's own padded-graph mask fix, 76b2e9d45e. Ours is not applied on
    # this container; this marker is what proves theirs arrived in its place.
    ('v1/attention/ops/common.py', 'sequence_indices = torch.searchsorted('),
    # ours, not upstream: hybrid + DCP + CPU offload
    ('v1/simple_kv_offload/manager.py', 'def _group_block_size'),
    ('v1/simple_kv_offload/manager.py', 'hit_length // self.block_size'),
]
for rel, marker in present:
    src = (root / rel).read_text()
    assert marker in src, f'DCP patch missing in {rel}: {marker}'

# Applying our cb810483 cudagraph fix here as well would double-patch the same
# function. It cannot be present.
common = (root / 'v1/attention/ops/common.py').read_text()
assert 'idx = torch.searchsorted(query_start_loc[1:], row, right=True)' not in common, (
    'our cb810483 cudagraph fix is present; on this container 76b2e9d45e supersedes it'
)

# The assert this whole arm exists to remove.
k3 = (root / 'models/kimi_k3/nvidia/mla.py').read_text()
assert 'does not support context parallelism.' not in k3, (
    'DCP patch applied but the blanket context-parallel assert is still in mla.py'
)
print('DCP patch (new container) verified in', root)
"

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
