#!/usr/bin/env bash
# The B200 DCP8 chain plus the stride/pointer probe.
# =============================================================================
# The index question is closed: kdaprobe2 watermarked to call 82,297 and every index and
# extent held. The third input to those addresses -- the strides -- has never been
# measured, and all eight are tl.constexpr, which is what Triton keys its kernel cache
# on. conv_state may also be a transposed view. See
# vllm-container-deps-k3-strideprobe.sh. No behaviour change.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-strideprobe.sh
