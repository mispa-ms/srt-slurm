#!/bin/bash
# Kimi-K3 aggregated PP>1: HF cache shim + the DSpark/PP aux-relay patch.
#
# Same base as vllm-container-deps-k3-hfshim.sh (used by every K3 AGG config),
# plus the patch that lets `method: dspark` run with pipeline_parallel_size > 1.
# Split into its own wrapper so the existing PP=1 configs keep their setup script
# byte-identical and stay comparable against earlier runs.
#
# Ordering matters only in that the HF shim must land before the server starts;
# the aux-relay patch touches installed vLLM source and is independent of it.

set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pp.sh
