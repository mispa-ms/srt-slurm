#!/usr/bin/env bash
# The B200 DCP8 chain plus the checkpoint-column bound.
# =============================================================================
# Tests whether the PP2 illegal memory access is a block-table column read past the
# end. See vllm-container-deps-k3-ckptcol-bounds.sh for the reasoning; in short,
# checkpoint_cols is derived from seq_lens_cpu_upper_bound and guarded only against
# being negative, and the resulting state_idx is what the faulting Triton kernel
# dereferences.
#
# Same image, same DCP8, same three direct flags, same Mooncake as its parent
# -sasrt-latest arm, which faulted after 230 of 838 requests.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-ckptcol-bounds.sh
