#!/usr/bin/env bash
# The 08/28 DSpark chain, plus the mamba group-id fix that capture needs.
# =============================================================================
# The first 08/28 attempt got every patch applied and then died in cudagraph capture on
# an upstream assertion: "expected 3 block tables, got 4". Two independently computed
# copies of mamba_group_ids, and #52388 added a call site that establishes them at
# different moments. See vllm-container-deps-k3-mambagroups.sh.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828-dspark.sh
bash /configs/patches/vllm-container-deps-k3-mambagroups.sh
