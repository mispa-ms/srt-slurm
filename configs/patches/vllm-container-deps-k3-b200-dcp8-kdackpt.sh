#!/usr/bin/env bash
# The B200 DCP8 chain plus the KDA checkpoint-grid bound.
# =============================================================================
# Tests the kernel CUDA_LAUNCH_BLOCKING named for the PP2 illegal memory access:
# _store_cache_checkpoints_kernel, whose grid comes from checkpoint_offsets while it
# indexes checkpoint.state_indices and query_start_loc unmasked. See
# vllm-container-deps-k3-kdackpt-bounds.sh for the full reasoning.
#
# Same image, same DCP8, same three direct flags, same Mooncake as its parent
# -sasrt-latest arm, which faulted after 230 of 838 requests.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-kdackpt-bounds.sh
