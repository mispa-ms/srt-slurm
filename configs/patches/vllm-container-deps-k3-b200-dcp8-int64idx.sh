#!/usr/bin/env bash
# The B200 DCP8 chain plus the int64 cast on state_idx.
# =============================================================================
# One line against the chain every other arm on this branch runs, so the only difference
# is the cast. See vllm-container-deps-k3-int64idx.sh for why the product overflows and
# why our own measurements read clean while it did.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-int64idx.sh
