#!/usr/bin/env bash
# The 2026-08-28 nightly (6f7df92a), no speculation.
# =============================================================================
# What this nightly already has, so the chain no longer carries it: mooncake #53324 and
# the cudagraph startup fix #53682. What it does not have, and needs:
#
#   pr54167     the low-latency GEMM mixin never calls its base initializer. #50572
#               landed here and #54167 merged two hours after the image was built, so
#               any shape the fast path declines raises AttributeError on _gemm_impl.
#   dcp8        the DCP dummy-batch fix, or every DP-idle step reads a
#               dcp_local_seq_lens the dummy path never set
#   int64idx    state_idx to int64, or the run dies at ~19.5 min
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-pr54167.sh
bash /configs/patches/vllm-container-deps-k3-int64idx.sh
