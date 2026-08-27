#!/usr/bin/env bash
# The B200 DCP8 chain plus the four-tensor checkpoint grid bound.
# =============================================================================
# The probe cleared the conv half -- its lower-bound flag was a row the kernel already
# masks -- and pointed at the half it never measured: checkpoint_state, read at
# seq_idx * checkpoint_stride_0 with nothing bounding seq_idx. The earlier clamp used
# three tensors and missed exactly this one, which is why it changed nothing.
#
# See vllm-container-deps-k3-ckptstate-bounds.sh. The column guard is kept; its
# violations are real and separate.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-ckptcol-bounds.sh
bash /configs/patches/vllm-container-deps-k3-ckptstate-bounds.sh
