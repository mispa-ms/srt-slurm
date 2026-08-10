# SPDX-License-Identifier: Apache-2.0
"""Is the hit length the lookup reports actually loadable from the store?

Run against the *installed* vllm inside the framework container, before any GPU
time, by kimi-k3-merged-mooncake.sh.

A Mooncake key is a whole block. The store therefore holds an object only at
each group's block boundaries -- there is no object at a hash boundary in a
block's interior. Core's fine-grained lookup deliberately extends a hit *into*
the first non-full block (a local block is usable as a prefix; a remote object
is not), and with EAGLE/DSpark it then subtracts exactly one hash unit, which
lands mid-block by construction.

Off DCP that gap is empty: the attention block equals hash_block_size. Scaling
the attention block by dcp opens a block_size/hash_block_size-wide interior, and
every hit landing there names a key nobody wrote. Measured on B300 c8 DCP=8:
2,757,664 OBJECT_NOT_FOUND (-704), all on the one scaled group, the run stuck in
warmup for four hours because kv_load_failure_policy=recompute retried forever.

vllm#50359 revalidates the reconciled boundary and steps back until the key
exists. This asserts the property that fix is for, over dcp in {1,2,4,8}: the
reported hit must be an exact object boundary for EVERY group, not just the one
upstream checks.

The gates that ran instead of this could not see it -- GSM8K passed at
0.950/0.954/0.956 with an external hit rate of 0.0%, never executing the
connector's read path at all.
"""

from math import lcm

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

HASH_BLOCK_SIZE = 128
MAMBA_BLOCK = 1536
# 64512 = 42 * 1536: a Mamba and hash boundary, but not a dcp-scaled attention
# one (64512 / 12288 = 5.25 at dcp=8). The defect is length-dependent, which is
# why one lucky length shows nothing.
MAX_LENGTH = 64512


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


def check(dcp: int) -> None:
    groups = _groups(dcp)
    block_sizes = [g.kv_cache_spec.block_size for g in groups]
    coord = MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=lcm(*block_sizes),
        hash_block_size=HASH_BLOCK_SIZE,
        use_eagle=True,
        retention_interval=0,
    )
    hashes = [
        BlockHash(i.to_bytes(4, "big")) for i in range(MAX_LENGTH // HASH_BLOCK_SIZE)
    ]

    # What the store actually holds: one object per group per block boundary.
    exists: set[tuple[int, bytes]] = set()
    for g_idx, block_size in enumerate(block_sizes):
        for end in range(block_size, MAX_LENGTH + 1, block_size):
            exists.add((g_idx, bytes(hashes[end // HASH_BLOCK_SIZE - 1])))

    _masks, hit = coord.find_longest_cache_hit(
        hashes,
        max_length=MAX_LENGTH,
        cached_block_pool=ExternalCachedBlockPool(HASH_BLOCK_SIZE, exists),
    )

    if hit == 0:
        return
    for g_idx, block_size in enumerate(block_sizes):
        key = bytes(hashes[hit // HASH_BLOCK_SIZE - 1])
        assert (g_idx, key) in exists, (
            f"dcp={dcp}: reported hit {hit} is not an object boundary for group "
            f"{g_idx} (block_size={block_size}); loading it asks Mooncake for a "
            f"key nobody wrote -> -704"
        )


def check_load_mask_not_shortened() -> None:
    """The recv-side pool answers "present" to anything, so the exact-boundary
    retry has no truth to check there. If it ran, load_mask would return a mask
    for a shorter length while its caller keeps using the original token_len,
    and process_tokens' trailing chunks would fall off the end of the mask --
    those blocks stay uninitialized in the local KV pool. Silent, unlike -704.
    """
    groups = _groups(8)
    block_sizes = [g.kv_cache_spec.block_size for g in groups]
    coord = MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=lcm(*block_sizes),
        hash_block_size=HASH_BLOCK_SIZE,
        use_eagle=True,
        retention_interval=0,
    )
    token_len = 61440  # 5 * 12288: block-aligned, as a validated hit must be
    hashes = [
        BlockHash(i.to_bytes(4, "big")) for i in range(token_len // HASH_BLOCK_SIZE)
    ]
    for block_size, mask in zip(block_sizes, coord.load_mask(hashes, token_len)):
        assert len(mask) == token_len // block_size, (
            f"load_mask shortened the group with block_size={block_size}: "
            f"{len(mask)} chunks for {token_len} tokens"
        )


def check_mamba_only_lookup() -> None:
    """Partial hits are enabled by a Mamba group alone, so the revalidation can
    be reached with no FullAttention group present."""
    groups = [_groups(1)[1]]
    coord = MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=MAMBA_BLOCK,
        hash_block_size=HASH_BLOCK_SIZE,
        use_eagle=True,
        retention_interval=0,
    )
    assert coord.enable_partial_hash_hits
    hashes = [BlockHash(i.to_bytes(4, "big")) for i in range(12)]
    coord.find_longest_cache_hit(
        hashes,
        max_length=MAMBA_BLOCK,
        cached_block_pool=ExternalCachedBlockPool(
            HASH_BLOCK_SIZE, {(0, bytes(hashes[11]))}
        ),
    )


def main() -> int:
    """Plain driver: the framework image is not guaranteed to ship pytest, and a
    missing test dependency must not take down every arm of a sweep."""
    failures = 0
    for dcp in (1, 2, 4, 8):
        try:
            check(dcp)
            print(f"  ok    reported hit is externally loadable [dcp={dcp}]")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  [dcp={dcp}]: {e}")
    for name, fn in (
        ("load_mask is not shortened by the retry", check_load_mask_not_shortened),
        ("mamba-only lookup does not crash", check_mamba_only_lookup),
    ):
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
    if failures:
        print(f"\n{failures} mooncake DCP hit-boundary check(s) failed.")
        return 1
    print("=== mooncake DCP hit-boundary tests passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
