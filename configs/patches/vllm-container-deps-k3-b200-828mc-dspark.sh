#!/usr/bin/env bash
# 08/28 DSpark + PP, with the minimal mamba group-id cache fix.
# =============================================================================
# See vllm-container-deps-k3-mambacache.sh.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828-dspark.sh
bash /configs/patches/vllm-container-deps-k3-mambacache.sh
