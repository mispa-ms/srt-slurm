#!/usr/bin/env bash
# 08/28 no-spec, with the minimal mamba group-id cache fix.
# =============================================================================
# Same as the 828 chain, ending in mambacache instead of mambagroups2: two lines that
# key the cache on the config it was derived from, rather than three call sites and a
# rewritten loop. See vllm-container-deps-k3-mambacache.sh.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828.sh
bash /configs/patches/vllm-container-deps-k3-mambacache.sh
