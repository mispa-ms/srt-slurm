#!/usr/bin/env bash
# The B200 DCP8 chain, the checkpoint-column guard, and the index probe.
# =============================================================================
# Diagnostic arm. Four fixes for the PP2 illegal memory access have been tried and all
# four were wrong; this one measures instead. See vllm-container-deps-k3-kdaprobe.sh.
#
# The column guard is kept because the out-of-range columns it reports are real and
# independently worth excluding, so anything the probe still flags is a second problem.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-ckptcol-bounds.sh
bash /configs/patches/vllm-container-deps-k3-kdaprobe.sh
