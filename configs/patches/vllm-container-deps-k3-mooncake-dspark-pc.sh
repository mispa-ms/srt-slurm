#!/usr/bin/env bash
# Kimi-K3 disagg: HF shim + Mooncake (bia) + Hanjie Qiu's DSpark/prefix-cache patches.
# =============================================================================
# Ordering matters and each step is idempotent:
#
#   1. k3-hfshim        resolve hf:moonshotai/Kimi-K3 to the staged 1.4 TB copy
#   2. mooncake         bia wheel + config + master (carries the free-port probe
#                       that fixed the master half-starting in bia's shared netns)
#   3. dspark-prefixcache-nixl   Hanjie's two container patches:
#                         - remove the DS-layout speculative-decoding assert in
#                           vllm/v1/worker/gpu/model_states/mamba_hybrid.py
#                         - generalise NIXL's SSM local-block assert from
#                           "exactly 1" to "1 committed + N speculative" slots
#   4. hybrid invalid-blocks recovery fix (optional file, from the same package)
#
# Not taken from that package: its mooncake-cuda13.sh, which swaps the wheel for
# a CUDA 13 build. bia is x86 and the container's own Mooncake imports fine here;
# our mooncake.sh is the path that produced a live store (Keys 1,252 / 24.76 GB).
#
# Source: kimi-k3-disagg-dspark-prefixcache-mooncake-package, validated on
# OCI-Aga GB300 jobs 326172 (c1) and 326173 (c4), 2026-07-30. The patch wrappers
# verify their anchors and fail closed if the container source has moved.
#
# CAVEAT: the removed assert guarded a real kernel assumption -- "the fused copy
# kernels shift conv windows assuming the SD layout; the DS layout cannot express
# a >0 spec-decode shift as a single contiguous copy". Its author states the
# patches are agent-generated to unblock benchmarks and are not quality-checked.
# Treat any accuracy claim from a run using this script as unverified.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-mooncake.sh
bash /configs/patches/vllm-container-deps-dspark-prefixcache-nixl.sh

if [ -f /configs/patches/vllm_hybrid_invalid_blocks_fix.py ]; then
    python3 /configs/patches/vllm_hybrid_invalid_blocks_fix.py
fi
