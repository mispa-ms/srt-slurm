#!/usr/bin/env bash
# The 08/28 no-spec chain, plus the mamba group-id fix that capture needs.
# =============================================================================
# See vllm-container-deps-k3-mambagroups.sh. The assertion fires during capture, so it
# hits the no-spec arm too -- speculation has nothing to do with it.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828.sh
bash /configs/patches/vllm-container-deps-k3-mambagroups.sh
