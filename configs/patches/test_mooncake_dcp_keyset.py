# SPDX-License-Identifier: Apache-2.0
"""Does a Mooncake load ever ask for a key the save path never wrote?

Run against the *installed* vllm inside the framework container, before any GPU
time, by kimi-k3-merged-mooncake.sh. A standalone copy of the tests added to
tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py, because the image
ships site-packages and not the test tree.

This is the check that was missing when the DCP + hybrid refusal was lifted. The
gates that ran instead -- patch applied, GSM8K within 1 SE, FULL cudagraphs N/N
-- all passed on a build that then livelocked with 2,757,664 OBJECT_NOT_FOUND
(-704) failures, because GSM8K's working set fits in GPU KV and never executed
the connector's read path (external hit rate 0.0%).
"""

from math import lcm

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
    partial_hash_hits_enabled,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

HASH_BLOCK_SIZE = 128
MAMBA_BLOCK = 1536
# 64512 = 42 * 1536: aligned to the Mamba block and to hash_block_size, but not
# to the dcp-scaled attention block (64512 / 12288 = 5.25 at dcp=8). The defect
# is length-dependent -- dcp=2 divides it evenly and shows nothing.
RAW_HIT = 64512


def _groups(dcp: int):
    """Kimi-K3's shape: one attention group scaled by dcp, one Mamba group not."""
    return [
        KVCacheGroupSpec(
            ["L0"],
            FullAttentionSpec(
                block_size=MAMBA_BLOCK * dcp,
                num_kv_heads=8,
                head_size=64,
                dtype=None,
                sliding_window=None,
            ),
        ),
        KVCacheGroupSpec(
            ["L1"],
            MambaSpec(
                block_size=MAMBA_BLOCK,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
    ]


def _keys(block_size, token_len, hashes):
    db = ChunkedTokenDatabase(
        KeyMetadata("m", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0),
        block_size,
        hash_block_size=HASH_BLOCK_SIZE,
    )
    return {bytes(h) for _, _, h in db.process_tokens(token_len, hashes)}


def test_load_never_requests_a_key_the_producer_did_not_write(dcp):
    groups = _groups(dcp)
    block_sizes = [g.kv_cache_spec.block_size for g in groups]
    coord = MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=lcm(*block_sizes),
        hash_block_size=HASH_BLOCK_SIZE,
        use_eagle=True,
        dcp_world_size=dcp,
    )
    hashes = [
        BlockHash(i.to_bytes(4, "big")) for i in range(RAW_HIT // HASH_BLOCK_SIZE)
    ]
    # The producer floors its save to lcm_block_size (KVCacheStoreSendingThread);
    # the consumer loads up to align_lookup_length(hit).
    saved = RAW_HIT // coord.lcm_block_size * coord.lcm_block_size
    loaded = coord.align_lookup_length(RAW_HIT)

    for block_size in block_sizes:
        missing = _keys(block_size, loaded, hashes) - _keys(block_size, saved, hashes)
        assert not missing, (
            f"dcp={dcp} block_size={block_size}: load would request "
            f"{len(missing)} key(s) the producer never wrote "
            f"(saved={saved}, loaded={loaded}) -> Mooncake -704"
        )


def test_partial_hash_hits_off_when_dcp_scales_an_attention_group(dcp):
    assert not partial_hash_hits_enabled(_groups(dcp), HASH_BLOCK_SIZE, dcp)
    # Still on without DCP: the fix must not change a path that already worked.
    assert partial_hash_hits_enabled(_groups(1), HASH_BLOCK_SIZE, 1)
    # Mamba-only is unaffected -- DCP does not scale it.
    mamba_only = [_groups(1)[1]]
    assert partial_hash_hits_enabled(mamba_only, HASH_BLOCK_SIZE, dcp)


def test_scheduler_and_worker_group_lists_agree(dcp):
    """The scheduler passes the raw config, the worker its DCP-scaled copy.

    A disagreement here is what makes the scheduler promise a hit length the
    worker cannot serve.
    """
    raw = _groups(1)
    scaled = _groups(dcp)
    assert partial_hash_hits_enabled(raw, HASH_BLOCK_SIZE, dcp) == (
        partial_hash_hits_enabled(scaled, HASH_BLOCK_SIZE, dcp)
    )


def main() -> int:
    """Plain driver: the framework image is not guaranteed to ship pytest, and a
    missing test dependency must not take down every arm of a sweep."""
    cases = [
        (test_load_never_requests_a_key_the_producer_did_not_write, (1, 2, 4, 8)),
        (test_partial_hash_hits_off_when_dcp_scales_an_attention_group, (2, 8)),
        (test_scheduler_and_worker_group_lists_agree, (2, 8)),
    ]
    failures = []
    for fn, dcps in cases:
        for dcp in dcps:
            try:
                fn(dcp)
                print(f"  ok    {fn.__name__}[dcp={dcp}]")
            except AssertionError as e:
                failures.append(f"{fn.__name__}[dcp={dcp}]: {e}")
                print(f"  FAIL  {fn.__name__}[dcp={dcp}]: {e}")
    if failures:
        print(f"\n{len(failures)} mooncake DCP key-set check(s) failed.")
        return 1
    print("=== mooncake DCP key-set tests passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
