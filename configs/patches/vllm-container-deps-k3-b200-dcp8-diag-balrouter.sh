#!/usr/bin/env bash
# The B200 DCP8 diag chain plus Wei's balanced-router diagnostic. MXFP4 arm.
set -euo pipefail
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-balrouter.sh
