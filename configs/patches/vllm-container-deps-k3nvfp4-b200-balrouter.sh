#!/usr/bin/env bash
# The NVFP4 B200 chain plus Wei's balanced-router diagnostic. NVFP4 arm.
set -euo pipefail
bash /configs/patches/vllm-container-deps-k3nvfp4-b200.sh
bash /configs/patches/vllm-container-deps-k3-balrouter.sh
