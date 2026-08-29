#!/usr/bin/env bash
# c24 ns=7 baseline, plus #51358 and #53598.
# =============================================================================
# WHY #53614 IS NOT HERE. The first attempt at this combination carried #53614 as well,
# and all four arms that included it died at its own assertion, ~11 minutes into a
# benchmark that had until then served normally:
#
#   vllm/v1/core/single_type_kv_cache_manager.py  _cache_partial_tail_block
#   assert not checkpoint_block.is_null
#   AssertionError -> EngineCore encountered a fatal error
#
# pipeline 65246264, children 65246269/70/72/73. Identical stack on all four. The
# assertion is added by #53614 and by nothing else in the chain -- neither #51358,
# #53598 nor k3-mooncake-53324-828 mentions it, and bare 6f7df92a does not have it.
# Applying #51358 first does not help: the three -pall arms had it and died the same
# way. So this is not the missing-dependency case its rebase suggested; #53614 does not
# hold on K3 + DCP8 + Mooncake align as it stands. It is an open PR.
#
# A second speculative concurrency so the c48 result is not a single point. c24 ns=7 is on the union frontier at 6,714.4 tok/s/chip @ x 45.23.
#
# The base chain is the one this arm's own baseline ran, so the only difference is the
# two patches. #51358 needs fuzz 2 on two hunks -- mooncake/store/scheduler.py #2 and
# v1/core/sched/scheduler.py #3 -- because k3-mooncake-53324-828 shifts their context.
# Verified at default fuzz on 6f7df92a+53324: 0 rejects, partial_tail_offloads= removed
# and kv_connector_block_state= inserted as the hunk intends, per-request state dicts
# still initialised, 16 files compile. Do NOT add --fuzz=0 here; it rejects both hunks
# and leaves a tree that imports and is wrong.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828-dspark3.sh
bash /configs/patches/vllm-container-deps-k3-pr51358.sh
bash /configs/patches/vllm-container-deps-k3-pr53598.sh
