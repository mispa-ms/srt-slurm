#!/usr/bin/env bash
# 08/28 DSpark + PP, with the corrected mamba group-id fix.
# =============================================================================
# See vllm-container-deps-k3-mambagroups2.sh for why v1 was wrong and what replaced it.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828-dspark.sh
bash /configs/patches/vllm-container-deps-k3-mambagroups2.sh
