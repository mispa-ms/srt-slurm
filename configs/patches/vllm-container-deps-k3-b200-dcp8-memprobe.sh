#!/usr/bin/env bash
# The B200 DCP8 chain plus the storage-bounds and workspace-identity probe.
# =============================================================================
# The stride arm came back with a single stride signature and no change, so strides are
# cleared for the window it watched -- but it capped at 60 reports per worker and went
# quiet at call 178, twenty minutes before the fault, which is the same blind spot the
# index probe had. Its pointer churn turned out to be layer identity, not reallocation:
# conv_state and recurrent_state are two views of one slab, always 27648 bytes apart,
# while the qkv and workspace bases barely move.
#
# What is left is the thing pointer arithmetic cannot answer -- whether those views fit
# inside their own storage -- and the workspace, which PP2 is what makes interesting.
# See vllm-container-deps-k3-memprobe.sh. No behaviour change.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-memprobe.sh
