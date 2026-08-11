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

import inspect
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


class Skipped(Exception):
    """The build under test does not carry the code a check is for."""


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


def check_replay_boundary_is_loadable(dcp: int) -> None:
    """The EAGLE replay-boundary fast path must obey the same object-boundary
    rule as the ordinary lookup.

    wzhao18/vllm@d87cdf5ce4 lets a fine-grained EAGLE producer resume directly
    at ``latest_boundary - hash_block_size`` instead of taking a lookahead
    snapshot. That boundary is a hash multiple by construction, so it lands in a
    dcp-scaled attention block's interior. It is loadable only because the
    producer's partial-tail offload wrote an object there; if it did not, or did
    so for only some groups, returning it is the -704 livelock again by a new
    path. The fast path is therefore gated on the same revalidation as the loop.

    The invariant is not "the hit is block aligned" -- under a partial-tail
    offload it deliberately is not -- but "the hit's key exists for every
    group", which is what a load actually requires.

    Two stores, because each alone is weak. With the tail written, the fast path
    fires and must return it; the run is asserted non-vacuous, since a store
    that never triggers the fast path would pass while testing nothing. With the
    tail written for only the Mamba group, the fast path must refuse it -- that
    is the case the gate exists for.
    """
    # d87cdf5ce4 is applied by the disagg setup script only, so on an AGG run
    # there is no fast path to exercise and the vacuity assert below would fire
    # with nothing wrong. Probe rather than assume.
    if "replay_boundary" not in inspect.getsource(
        MooncakeStoreCoordinator.find_longest_cache_hit
    ):
        raise Skipped("no EAGLE replay fast path in this build")

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

    def store(tail_groups: tuple[int, ...]) -> set[tuple[int, bytes]]:
        exists: set[tuple[int, bytes]] = set()
        for g_idx, block_size in enumerate(block_sizes):
            for end in range(block_size, MAX_LENGTH + 1, block_size):
                exists.add((g_idx, bytes(hashes[end // HASH_BLOCK_SIZE - 1])))
        for length in range(MAMBA_BLOCK, MAX_LENGTH + 1, MAMBA_BLOCK):
            replay = length // HASH_BLOCK_SIZE * HASH_BLOCK_SIZE - HASH_BLOCK_SIZE
            if replay > 0:
                for g_idx in tail_groups:
                    exists.add((g_idx, bytes(hashes[replay // HASH_BLOCK_SIZE - 1])))
        return exists

    all_groups = tuple(range(len(block_sizes)))
    for label, tail_groups in (("all groups", all_groups), ("mamba only", (1,))):
        exists = store(tail_groups)
        pool = ExternalCachedBlockPool(HASH_BLOCK_SIZE, exists)
        fired = 0
        for max_length in range(MAMBA_BLOCK, MAX_LENGTH + 1, MAMBA_BLOCK):
            _masks, hit = coord.find_longest_cache_hit(
                hashes, max_length=max_length, cached_block_pool=pool
            )
            if hit == 0:
                continue
            replay = max_length // HASH_BLOCK_SIZE * HASH_BLOCK_SIZE - HASH_BLOCK_SIZE
            fired += hit == replay
            key = bytes(hashes[hit // HASH_BLOCK_SIZE - 1])
            for g_idx, block_size in enumerate(block_sizes):
                assert (g_idx, key) in exists, (
                    f"dcp={dcp} max_length={max_length} tail={label}: hit {hit} "
                    f"names a key group {g_idx} (block_size={block_size}) never "
                    f"wrote -> -704"
                )
        # Only the positive store proves the path is live. In the mamba-only
        # store an ungated fast path is caught by the key-exists assertion
        # above, so there is nothing further to assert here.
        if tail_groups == all_groups:
            assert fired, (
                f"dcp={dcp}: the EAGLE replay fast path never fired even with the "
                f"partial tail written for every group -- this check is vacuous "
                f"and would pass with the revalidation gate removed"
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
        try:
            check_replay_boundary_is_loadable(dcp)
            print(f"  ok    EAGLE replay boundary is loadable [dcp={dcp}]")
        except Skipped as e:
            print(f"  skip  EAGLE replay boundary [dcp={dcp}]: {e}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  replay boundary [dcp={dcp}]: {e}")
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
