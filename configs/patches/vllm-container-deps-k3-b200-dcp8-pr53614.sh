#!/usr/bin/env bash
# The B200 DCP8 chain plus upstream PR #53614.
# =============================================================================
# One patch per arm. The DCP8 chain is what every other arm on this branch runs, so the
# only difference against them is #53614. See vllm-container-deps-k3-pr53614.sh.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-pr53614.sh
