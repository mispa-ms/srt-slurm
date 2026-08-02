# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make external KV-load failures non-fatal for hybrid models."""

import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
MARKER = "req_hybrid_block_ids = {"

OLD = """            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])"""

NEW = """            # get_block_ids returns one block-id list per KV-cache group.
            # Hybrid models (e.g. Mamba + attention) expose several groups; in
            # align mode they share the scheduler block_size and block count,
            # and block_ids are globally unique. Scan by logical position and
            # treat a position as invalid if ANY group's block there failed to
            # load, truncating the request at the earliest such position.
            # (Previously this unpacked a single group -- ``(req_block_ids,) =
            # ...`` -- and crashed hybrid models with a failed external KV load
            # via "ValueError: too many values to unpack (expected 1)".)
            block_ids_per_group = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            if len(block_ids_per_group) != 1:
                # Hybrid: attention and Mamba groups share no token boundary.
                # mamba-cache-mode align equalises page *bytes*, not token
                # counts, and mooncake reports failures as physical block IDs
                # with no group attached, so idx * block_size is a position that
                # is only meaningful for one group. Truncating there reloads the
                # same blocks and fails identically -- 60586688 rescheduled two
                # requests 1,959 and 1,955 times with `tokens affected` frozen.
                # Drop the whole prefix instead: expensive, but it converges.
                req_hybrid_block_ids = {
                    block_id
                    for group_block_ids in block_ids_per_group
                    for block_id in group_block_ids
                }
                if req_hybrid_block_ids.isdisjoint(invalid_block_ids):
                    continue
                is_affected = True
                marked_invalid_block_ids.update(
                    req_hybrid_block_ids & invalid_block_ids
                )
                request.num_computed_tokens = 0
                total_affected_tokens += max(req_num_computed_tokens, 0)
                if evict_blocks:
                    blocks_to_evict.update(req_hybrid_block_ids)
                continue

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx in range(req_num_computed_blocks):
                invalid_here = [
                    group_block_ids[idx]
                    for group_block_ids in block_ids_per_group
                    if idx < len(group_block_ids)
                    and group_block_ids[idx] in invalid_block_ids
                ]
                if not invalid_here:
                    continue

                is_affected = True

                # A position is "shared" only if every invalid block at it was
                # already marked by a previous request (which will recompute
                # it), so this request can still treat it as computed.
                new_invalid = [
                    block_id
                    for block_id in invalid_here
                    if block_id not in marked_invalid_block_ids
                ]
                marked_invalid_block_ids.update(invalid_here)

                if not new_invalid:
                    # All invalid blocks here are shared with a previous request
                    # and already marked for recomputation.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing.
                    continue

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks,
                # across every group
                if evict_blocks:
                    for group_block_ids in block_ids_per_group:
                        blocks_to_evict.update(group_block_ids[idx:])"""


def main():
    if not TARGET.exists():
        print(
            f"[vllm-hybrid-invalid-blocks-fix] Target not found: {TARGET}",
            file=sys.stderr,
        )
        sys.exit(1)

    content = TARGET.read_text()

    if MARKER in content:
        print(
            "[vllm-hybrid-invalid-blocks-fix] Already patched, skipping.",
            file=sys.stderr,
        )
        return

    count = content.count(OLD)
    if count == 0:
        print(
            "[vllm-hybrid-invalid-blocks-fix] Could not find the single-group "
            "unpack anchor. vLLM version may have drifted; inspect "
            "Scheduler._update_requests_with_invalid_blocks().",
            file=sys.stderr,
        )
        sys.exit(1)
    if count > 1:
        print(
            f"[vllm-hybrid-invalid-blocks-fix] Anchor is ambiguous ({count} "
            "occurrences); refusing to patch.",
            file=sys.stderr,
        )
        sys.exit(1)

    content = content.replace(OLD, NEW)
    TARGET.write_text(content)
    print(
        "[vllm-hybrid-invalid-blocks-fix] Rewrote "
        "_update_requests_with_invalid_blocks to handle hybrid multi-group KV "
        "(hybrid requests drop the whole prefix and recompute, which converges).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
