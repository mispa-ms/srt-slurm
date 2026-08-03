#!/usr/bin/env bash
# Kimi-K3 disagg on the kimi-k3 image, with SemiAnalysis's three vLLM patches
# in place of our own.
# =============================================================================
# Successor to vllm-container-deps-k3-mooncake-dspark-pc.sh. Two of that
# script's three patches were upstreamed by vLLM PR #50000; the third
# (vllm_hybrid_invalid_blocks_fix.py) is dropped here in favour of SA's, which
# fixes the same function differently:
#
#   ours  scan the groups and truncate at the earliest invalid logical position
#   SA    if the request has more than one KV group, recompute the whole prefix
#
# SA's comment states why ours cannot be right: "Attention and Mamba groups can
# use different logical block sizes, so there is no safe common token boundary
# at which to truncate them." Ours assumes such a boundary exists and cuts at
# idx x 1536, which lands correctly for the attention group and not for the
# mamba group -- the same block fails again on retry and recovery never
# advances. Expensive but always-progressing beats cheap and stuck.
#
# The three patches, taken from SemiAnalysisAI/InferenceX branch
# agent/kimi-k3-gb300-agentx (PR #2444, closed unmerged after ~20 CI iterations
# -- field notes, not a validated recipe):
#
#   patch_kimi_k3_v2_ds_prefix_cache.py
#       Backports vLLM #49291 (V2 DS align-state enablement) and #50153 (NIXL
#       multi-slot SSM pairing) into the July 27 kimi-k3 image.
#
#   patch_kimi_k3_mooncake_hma_recompute.py
#       v1/core/sched/scheduler.py. Conservative full-prefix recompute when a
#       failed external KV load cannot be attributed to a hybrid group.
#
#   patch_kimi_k3_mooncake_save_groups.py
#       mooncake/store/worker.py. Rejects a decode-side save whose block-table
#       tuple is missing hybrid groups, which a full external hit can produce.
#       This is the candidate fix for our 1P+1D c8 crash: prefill died on
#       `assert len(new_computed_blocks) == 0` in single_type_kv_cache_manager
#       after mooncake served an entire prefix (pipeline 60772697). It gets more
#       likely the more the store is read, and our load_get went from 584 to
#       3,696 once lookup_async was turned off.
#
# Not taken from SA: their vllm-container-deps.sh and the CUDA 13 Mooncake
# reinstall. bia is x86 and our mooncake path is already proven here -- 2.49 TB
# read across 3,696 load_get on pipeline 60758459.
#
# THIS VARIANT adds the fourth patch, a backport of vLLM PR #45406, which is the
# one that targets the warmup hang directly:
#
#   patch_vllm_45406_unstrand_parked_kv_loads.py
#       v1/core/sched/scheduler.py. When allocation fails for the head of the
#       waiting queue AND nothing is running, keep scanning instead of breaking,
#       so requests parked in WAITING_FOR_REMOTE_KVS behind it get promoted and
#       can release their blocks. #45406 is open since 2026-06-12 and has never
#       been merged; issue #45387 was closed by the reporter, not by a fix.
#
# Historical note kept for context: the warmup hang. Four ladder runs
# (c16/c24/c32/c40, pipelines 60773518 and 60774289) froze with prefill showing
# "Waiting: N reqs, Deferred: N reqs", errors=0, until the walltime. No assert,
# no engine death. lookup_async: false did not prevent it.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-mooncake.sh

python3 /configs/patches/patch_kimi_k3_v2_ds_prefix_cache.py
python3 /configs/patches/patch_kimi_k3_mooncake_hma_recompute.py
python3 /configs/patches/patch_kimi_k3_mooncake_save_groups.py
python3 /configs/patches/patch_vllm_45406_unstrand_parked_kv_loads.py
