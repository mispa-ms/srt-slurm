#!/usr/bin/env bash
# Kimi-K3 + Mooncake KV store.
# =============================================================================
# Thin wrapper: K3 needs the HF cache wired to the pre-staged 1.5 TB checkpoint,
# and Mooncake needs its wheel, config and master process. Both already exist as
# scripts, so this only orders them -- hfshim first, because the model has to
# resolve before anything else matters.
#
# vllm-container-deps-mooncake.sh carries the bia fix: mooncake_master's cinatra
# admin/metrics server could not bind (9003 default and our hard-coded 29003 were
# both taken in bia's shared netns) so the master half-started and died. It now
# probes for free ports. That was the real blocker, not the RDMA ibv_reg_mr
# failures an earlier diagnosis blamed.
#
# Both scripts source vllm-container-deps.sh themselves, so base deps run twice;
# apt/pip are idempotent and it costs a few seconds.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-mooncake.sh
