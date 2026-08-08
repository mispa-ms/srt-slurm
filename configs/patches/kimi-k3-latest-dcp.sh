#!/bin/bash
# The DCP stack rebuilt on a current nightly, for the second track.
#
# WHY A SECOND TRACK. The pinned track runs
# vllm/vllm-openai:nightly-cb8104839c... because #50484 and #50493 had their
# merge-base there. Three days on, that pin is far enough behind main that two
# things upstream changed under us, and both matter:
#
#   #50911  [Spec Decode] Enable fused non-causal TokenSpeed MLA for DSpark
#           adds supports_non_causal_multi_token_decode=True to
#           TokenspeedMLAMetadataBuilder. On the pinned image FLASHINFER_MLA was
#           the ONLY backend declaring it, so the DSpark draft was forced onto a
#           backend that does not declare supports_dcp_with_varlen -- which
#           clamps reorder_batch_threshold to 1 under DCP and killed capture.
#           Here the draft can stay on TOKENSPEED_MLA (threshold 5), so the
#           config needs NO backend split at all.
#
#   #50613  [Attention][MLA] Per-request scheduling for MLA chunked context
#           rewrote 726 lines of mla_attention.py. Our old
#           k3-dcp-pr50484-50493.patch does not apply across it (FAILED=9).
#
# WHAT THIS CARRIES, and why it is not the old patch regenerated. GirasoleY has
# since merged main into the PR branch, so the conflict #50613 caused is already
# resolved by the author. These are the PR's own commits as they stand on
# refs/pull/50484/head and refs/pull/50493/head, cherry-picked onto the nightly:
#
#   b91e26c9b8  [Attention] Direct symmetric-memory DCP A2A for MLA
#   e4f5666b77  [Kimi-K3] DCP support for the fused MLA layer
#   3dd7e92dfb  [Kimi-K3] Direct symmetric-memory DCP q-gather for MLA decode
#   5527c08894  DCP: publish query directly into final NVLS buffer
#   167b24ef99  Fix Kimi K3 fused MLA cache test arguments
#   76b2e9d45e  Fix DCP empty-shard masking for padded graphs
#   3120e9980f  [Core] Refactor cached block event emission
#   bcbb9e26bb  [Core] Report exact partial-prefix cache residency
#   9f409e3e2c  [Core] Support partial prefix cache hits under DCP
#
# All nine cherry-pick clean onto c810e5ee, and the resulting patch applies to a
# fresh checkout of it with zero fuzz.
#
# 76b2e9d45e IS OUR k3-dcp-cudagraph-pad-fix, upstreamed. That patch is
# deliberately NOT carried here; carrying both would conflict.
#
# STILL CARRIED, because upstream has not taken them:
#   k3-dcp-offload-hybrid-fix.patch    _group_block_size + the partial-hash-hit
#                                      floor. Xin Li's 2026-07-29 report is
#                                      still open and manager.py is unchanged.
#   k3-mamba-blocktable-dcp-fix        gpu/model_runner.py sizes EVERY group's
#                                      block table with cdiv(max_model_len,
#                                      block_size * dcp_size), applying the DCP
#                                      divisor to Mamba groups too. Mamba state
#                                      is replicated across DCP ranks and never
#                                      sharded. At DCP=8 the spec asks for 683
#                                      columns and the runner allocates 86, and
#                                      _get_aligned_state_indices_kernel bounds
#                                      only the row -- so past 86 * 1536 =
#                                      132,096 tokens it reads off the end. This
#                                      track's median ISL is ~104k and AgentX
#                                      sessions accumulate, so DCP=8 needs it.
#                                      At DCP=4 the same arithmetic gives 171
#                                      columns = 262,656 tokens.
#   wzhao-d87cdf5ce4                   still a prerequisite under speculation:
#                                      kv_cache_manager still carries
#                                      `assert num_blocks <= len(group_blocks)`,
#                                      which a spec arm reaches and no-spec does
#                                      not. Applies with fuzz 2 in MambaManager;
#                                      verified all three
#                                      cache_speculative_replay_tail sites land.
#
# The direct symmetric-memory CUDA kernels are still not compiled into the
# image, so VLLM_USE_DIRECT_DCP_{A2A,Q_GATHER,KV_GATHER} must still be 0. The
# check at the end of this script enforces it.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5.sh

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

apply_patch() {
    local name="$1" file="/configs/patches/$1.patch"
    echo "=== ${name} ==="
    if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$file" >/dev/null 2>&1; then
        echo "${name} already applied"
    else
        patch -p1 --forward -d "$SITE" < "$file"
    fi
}

apply_patch k3-dcp-pr50484-50493-latest
apply_patch k3-dcp-offload-hybrid-fix
apply_patch wzhao-d87cdf5ce4-partial-prefix-hits
apply_patch k3-mamba-blocktable-dcp-fix

echo "=== hybrid-KV recompute patch ==="
python3 /configs/patches/patch_kimi_k3_mooncake_hma_recompute.py || echo "NOTE: hma recompute patch not applied"

python3 -c "
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent

present = [
    ('models/kimi_k3/nvidia/mla.py', 'self.dcp_manager'),
    ('v1/attention/ops/dcp_utils.py', 'class MLADCPManager'),
    ('v1/attention/ops/common.py', 'def mask_dcp_empty_shards_'),
    ('envs.py', 'VLLM_USE_DIRECT_DCP_A2A'),
    # ours, not upstream
    ('v1/simple_kv_offload/manager.py', 'def _group_block_size'),
    # d87
    ('v1/core/single_type_kv_cache_manager.py', 'cache_speculative_replay_tail'),
    ('v1/core/kv_cache_coordinator.py', 'eagle_group_ids'),
    # Mamba groups must not take the DCP divisor on the block table
    ('v1/worker/gpu/model_runner.py', 'kv_shard_count'),
    ('models/kimi_k3/nvidia/kda_metadata.py', '_check_block_table_width'),
    # #50911: the draft no longer has to leave TOKENSPEED_MLA
    ('v1/attention/backends/mla/tokenspeed_mla.py',
     'supports_non_causal_multi_token_decode: ClassVar[bool] = True'),
]
for rel, marker in present:
    assert marker in (root / rel).read_text(), f'missing in {rel}: {marker}'

# d87's flag is wired at three sites; the patch applies with fuzz here, so check
# it landed whole rather than trusting the exit code.
n = sum(
    (root / r).read_text().count('cache_speculative_replay_tail')
    for r in ('v1/core/kv_cache_coordinator.py', 'v1/core/single_type_kv_cache_manager.py')
)
assert n == 3, f'd87 partially applied: {n} cache_speculative_replay_tail sites, expected 3'

print('latest DCP stack verified in', root)
"

for v in VLLM_USE_DIRECT_DCP_A2A VLLM_USE_DIRECT_DCP_Q_GATHER VLLM_USE_DIRECT_DCP_KV_GATHER; do
    val="${!v-}"
    if [ "$val" != "0" ]; then
        echo "FATAL - $v is '${val:-<unset>}', must be 0. The direct DCP CUDA kernels"
        echo "are not built into this image."
        exit 1
    fi
done
echo "direct DCP ops disabled (A2A/Q_GATHER/KV_GATHER = 0), using the Triton/NCCL fallback"
