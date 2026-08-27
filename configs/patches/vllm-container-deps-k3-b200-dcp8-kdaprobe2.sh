#!/usr/bin/env bash
# The B200 DCP8 chain plus the watermark index probe.
# =============================================================================
# The first index probe capped at 40 reports per worker and hit that cap on every one,
# so its "all indices in range" verdict describes the first seconds of serving, not the
# state around request 230 where the fault lands. This one watermarks instead, and adds
# the two cross-tensor size comparisons nobody has made. See
# vllm-container-deps-k3-kdaprobe2.sh. No behaviour change.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-kdaprobe2.sh
