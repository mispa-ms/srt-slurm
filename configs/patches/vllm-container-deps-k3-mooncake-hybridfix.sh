#!/usr/bin/env bash
# Kimi-K3 disagg on a stock vLLM main nightly: HF shim + Mooncake (bia) +
# the one Hanjie-package fix that is still not upstream.
# =============================================================================
# Successor to vllm-container-deps-k3-mooncake-dspark-pc.sh. K3 landed on vLLM
# main in PR #50000 (merge aeeb36b1f, 2026-07-30), which made two of that
# script's three patches unnecessary:
#
#   remove-vllm-ds-layout-spec-assert.patch  -> mamba_hybrid.py
#       DROPPED. The patch deleted an assert; main implements the behaviour it
#       guarded. _copy_mamba_state_block now applies the
#       token_bias = num_accepted - 1 window shift per conv layout (SD:
#       contiguous slice, DS: per-dim-row strided slice). This also retires the
#       accuracy caveat that applied to every run using the old script.
#
#   vllm-nixl-ssm-spec-prefix-cache.patch    -> nixl/base_worker.py
#       DROPPED. The "SSM can only have one local block" assert is gone;
#       transfers go through derive_mamba_conv_split (3-read Mamba conv
#       transfer). VLLM_SSM_CONV_STATE_LAYOUT=DS is still required and is
#       asserted by base_worker itself.
#
#   vllm_hybrid_invalid_blocks_fix.py        -> v1/core/sched/scheduler.py
#       KEPT. Still not upstream. main carries the original 44-line block
#       verbatim, including "# TODO (davidb): add support for hybrid memory
#       allocator" and the single-group unpack
#       "(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)".
#       K3 exposes several KV-cache groups (MLA + KDA), so a failed external KV
#       load raises "ValueError: too many values to unpack (expected 1)". That
#       is exactly our path: MooncakeStoreConnector with
#       kv_load_failure_policy=recompute.
#
# Verified against main on 2026-07-31: the patch's OLD anchor matches and its
# post-patch MARKER is absent. The wrapper fails closed if that stops holding.
#
# Intended container: vllm/vllm-openai:nightly-0f17394564fa2fccd332cf63321314884c15ee37
# (2026-07-31, 46 commits past the K3 merge; carries the K3 deepgemm fix #50458
# and the K3 tool-call-ID fix #50420, but not DSpark AR fusion #50242).
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-mooncake.sh

python3 /configs/patches/vllm_hybrid_invalid_blocks_fix.py
