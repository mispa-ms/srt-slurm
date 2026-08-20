#!/usr/bin/env bash
# Hanjie's strict port + vllm#50359 + the load-failure scan.
# =============================================================================
# The scan rides along on purpose. If #50359 takes the failures to zero the
# diagnostic prints nothing and costs nothing; if any survive, their group and
# chunk position are already in the log and no second run has to be queued.
# =============================================================================
set -euo pipefail

bash /configs/patches/kimi-k3-hanjie-strict.sh
bash /configs/patches/apply-vllm-load-failure-scan.sh
bash /configs/patches/apply-vllm-pr50359.sh
