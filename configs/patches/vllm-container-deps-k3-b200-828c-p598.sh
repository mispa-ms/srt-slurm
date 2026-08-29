#!/usr/bin/env bash
# c96 no-spec baseline, plus vLLM PR #53598 only.
# =============================================================================
# WHY C96 IS BEING RUN A THIRD TIME. The first two attempts both died, and each died at
# an assertion belonging to a different patch in the chain:
#
#   -pall (65246273)  #53614's own assertion, ~11 min in
#                     single_type_kv_cache_manager.py _cache_partial_tail_block
#                     assert not checkpoint_block.is_null
#
#   -p2   (65251566)  #51358's own assertion, after only 48 served requests
#                     mooncake/store/scheduler.py _apply_current_save_block_ids
#                     assert block_ids is not None
#                     "Missing current block table for store request chatcmpl-..."
#
# So #53598 has never actually been measured at c96: both times something else in the
# chain killed the arm first. This runs it alone.
#
# #51358 IS NOT SIMPLY BROKEN -- it is broken here. The same patch is alive at c48 ns=4
# (65246275, past 100 minutes) and in the c48/c24 -p2 arms. It fails at c96 no-spec,
# which is precisely the arm where the Mooncake catch was measured to collapse from
# 58.2% at c32 to 8.8%, with 4.23 TB evicted against a 3.12 TB store. An assertion that
# a request being stored has no current block table is the shape of a failure you would
# expect when the store is churning 1.56x its own contents. Worth reporting upstream:
# unlike #53614, #51358 is merged (2026-08-29T02:40Z).
#
# WHY #53598 IS THE ONE LEFT. It is the only patch of the three that has not failed
# anywhere, it is the smallest (two core files, +37/-10), and its mechanism is the most
# specific to this configuration: before it, replicated Mamba groups took the process
# DCP size for block geometry and prefix hashing, as if DCP8 had sharded state that is
# in fact per-rank. c96 is also where our margin over MI355X ATOM is thinnest -- +4.1%
# at their x 5.0 -- so it is the point most worth recovering.
#
# Base chain is 828c, the one the c96 baseline (8,121.2 tok/s/chip @ x 8.03) ran, so the
# only difference is the patch.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828c.sh
bash /configs/patches/vllm-container-deps-k3-pr53598.sh
