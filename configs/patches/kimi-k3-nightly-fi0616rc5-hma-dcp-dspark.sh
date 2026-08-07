#!/bin/bash
# kimi-k3-nightly-fi0616rc5-hma-dcp.sh plus DCP under DSpark speculative decoding.
#
# One delta against the script it wraps, so a no-spec DCP4 arm and a
# DSpark+DCP4 arm differ by exactly this patch plus the speculative-config
# and the target attention backend.
#
# WHAT IT CARRIES. k3-dcp-dspark-fix.patch, four changes, all ours:
#
#   1. config/speculative.py drops the K3DSparkModel-with-dcp>1 ValueError.
#      Base vLLM, predating the DCP PRs. It is a config-time check, so it fires
#      before the model is built and before anything else here is reachable.
#
#   2. models/kimi_k3/nvidia/mla.py drops the rotary_emb-is-None-under-dcp
#      assert. Its stated reason -- "gathered queries require gathered
#      positions" -- does not hold: RoPE is applied inside _decode_concat_cache
#      and the fused prefill key-concat, both BEFORE MLADCPManager.query_gather,
#      and that gather is all_gather(dim=1) over heads, not tokens. The prefill
#      context path takes no positions argument at all.
#
#   3. The DFlash/DSpark prepare-inputs kernel computed context and query slots
#      as block_id * block_size + pos % block_size with no CP awareness, while
#      resolve_kv_cache_block_sizes scales the draft group by dcp because
#      MLAAttentionSpec is a FullAttentionSpec. Silent wrong KV, not a crash --
#      the same shape as k3-dcp-offload-hybrid-fix part 2. EAGLE3 escaped this
#      only because the autoregressive speculator calls
#      block_tables.compute_slot_mappings. The fix extracts the interleave math
#      into one triton device function and routes both callers through it. The
#      block index divisor also moves from block_size to block_size * CP_SIZE.
#
#   4. The draft never received dcp_local_seq_lens. MLACommonMetadataBuilder
#      does `seq_lens = dcp_local_seq_lens` with no None guard when dcp > 1, so
#      the draft died on NoneType subscripting at its first forward.
#
# THE BACKEND SPLIT IS IN THE CONFIG, NOT HERE, and it is not optional. Under
# DCP, FlashInferMLAMetadataBuilder does not declare supports_dcp_with_varlen,
# so _init_reorder_batch_threshold forces reorder_batch_threshold to 1. The
# target's verify block is ns+1 tokens, and split_decodes_and_prefills then
# returns (0, num_reqs, 0, num_tokens) -- every request reclassified as
# prefill. No exception; every decode step runs the chunked-prefill path. The
# arm would measure as "works, but slow". So the target must be TOKENSPEED_MLA,
# which declares supports_dcp_with_varlen=True and passes causal_seqs / cp_world
# / cp_rank to its kernel. That is NOT a flashinfer bug: trtllm-gen's MLA decode
# takes no CP arguments and cannot express causal masking within a query block
# against an interleaved shard.
#
# The draft wants the opposite: supports_non_causal_multi_token_decode is
# declared on FlashInferMLAMetadataBuilder only. So the config sets
#
#   attention-backend: TOKENSPEED_MLA
#   speculative-config: {..., "attention_backend": "FLASHINFER_MLA", ...}
#
# This script cannot check that -- the backend is a CLI flag, not an env var --
# so the check is a hard gate on the worker log instead.
#
# NOT SETTLED HERE: prefix-match-unit under DSpark. use_eagle() returns True for
# "dspark", so vllm#51295 is live for this arm where it is a no-op for the
# no-spec ladder. It rewrites kv_cache_coordinator.py, which #50493 also
# rewrites, so it cannot simply be stacked. The gate does not depend on it -- a
# prefix-cache miss costs throughput, not correctness -- but the performance
# ladder does. See DCP_DSPARK_DESIGN.md section 7.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5-hma-dcp.sh

echo "=== k3-dcp-dspark-fix (ours, on top of the DCP stack) ==="
FIX_FILE=/configs/patches/k3-dcp-dspark-fix.patch
SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$FIX_FILE" >/dev/null 2>&1; then
    echo "k3-dcp-dspark-fix already applied"
else
    patch -p1 --forward -d "$SITE" < "$FIX_FILE"
fi

python3 -c "
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent

# Removals.
spec = (root / 'config/speculative.py').read_text()
assert 'MLA DSpark does not currently support decode context' not in spec, (
    'the DSpark x DCP config guard is still present'
)
mla = (root / 'models/kimi_k3/nvidia/mla.py').read_text()
# Note this one only means something after #50484 has landed -- the assert is
# introduced by that PR, so it is absent on a bare nightly too. -dcp.sh gates
# #50484 itself before we get here, which is what makes this check meaningful.
assert 'does not support RoPE with decode' not in mla, (
    'the RoPE x DCP assert is still present'
)
# PCP must stay refused.
assert 'does not support prefill context' in mla, (
    'the prefill-CP assert was removed; only the decode-CP one should be gone'
)

# Additions.
present = [
    ('v1/worker/gpu/cp_utils.py', 'def cp_local_slot'),
    ('v1/worker/gpu/block_table.py', 'cp_local_slot('),
    ('v1/worker/gpu/spec_decode/dflash/speculator.py', 'cp_local_slot('),
    ('v1/worker/gpu/spec_decode/dflash/speculator.py', 'prepare_dcp_local_seq_lens('),
    ('v1/worker/gpu/spec_decode/speculator.py', 'dcp_local_seq_lens'),
]
for rel, marker in present:
    assert marker in (root / rel).read_text(), f'dspark DCP fix missing in {rel}: {marker}'

# The block index divisor is the half of fix 3 that is easy to lose in a merge:
# without it the slot is CP-aware but indexes the wrong block table row.
dflash = (root / 'v1/worker/gpu/spec_decode/dflash/speculator.py').read_text()
assert dflash.count('// (block_size * CP_SIZE)') == 2, (
    'expected both dflash slot sites to index with block_size * CP_SIZE'
)

print('DSpark DCP patch verified in', root)
"
