#!/usr/bin/env bash
# 09/09 nightly + the two open-PR stopgaps + a runtime cherry-pick of vllm#55499.
# = vllm-container-deps-k3-b200-909min.sh + vllm-container-deps-k3-pr55499.sh
# A/B against -909min on the same image: the difference is the TRTLLM ragged prefill
# per-layer sync that FlashInfer 0.6.18 introduced and #55499 removes.
set -euo pipefail
echo "=== k3-b200-909min-55499 ==="
bash /configs/patches/vllm-container-deps-k3-b200-909min.sh
bash /configs/patches/vllm-container-deps-k3-pr55499.sh
echo "=== k3-b200-909min-55499: done ==="
