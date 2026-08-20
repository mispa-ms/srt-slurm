#!/usr/bin/env bash
# Hanjie's strict port, plus the mooncake load-failure scan diagnostic.
# =============================================================================
# Chains rather than forks: kimi-k3-hanjie-strict.sh runs unmodified, so this
# arm differs from the baseline by exactly one applied patch and two env vars.
# Anything else would have to be priced separately.
#
# The diagnostic is logging-only. See apply-vllm-load-failure-scan.sh for why
# it exists and what it prints.
# =============================================================================
set -euo pipefail

bash /configs/patches/kimi-k3-hanjie-strict.sh
bash /configs/patches/apply-vllm-load-failure-scan.sh
