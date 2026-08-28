#!/usr/bin/env bash
# The B200 DCP8 chain plus the in-kernel index check.
# =============================================================================
# memprobe closed the argument list: storage bounds, row contiguity and workspace
# identity are all correct, on top of indices (kdaprobe2) and strides (strideprobe).
# Every one of those checks read a device tensor on the host, before an asynchronous
# launch -- but this kernel loads state_idx and checkpoint_offset itself, at execution
# time, so none of that evidence describes the moment the fault happens.
#
# It also has no upper bound on either index: state_idx is tested against the sentinel
# and nothing else, and token_idx against nothing at all.
#
# This arm counts both on the device and adds the missing bounds. See
# vllm-container-deps-k3-kprobe.sh for how to read the four outcomes.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-kprobe.sh
