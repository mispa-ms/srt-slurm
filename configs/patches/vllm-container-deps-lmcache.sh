#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LMCache CPU KV-offload setup for vLLM agg (AgentX).
# Mirrors InferenceX benchmarks/single_node/agentic/kimik2.5_fp4_b300.sh `lmcache` case:
#   OFFLOAD_ARGS=(--kv-offloading-backend lmcache --kv-offloading-size $GB --disable-hybrid-kv-cache-manager)
#   + LMCACHE_LAZY_MEMORY_INITIAL_RATIO/STEP_RATIO env.
# vLLM ships the lmcache_integration adapter but NOT the lmcache runtime package, so install it.

set -euo pipefail

# Base container deps first (numactl, msgpack, numa-bind fix, host-mem poller).
bash /configs/patches/vllm-container-deps.sh

# ---- LMCache runtime install -------------------------------------------------
# vllm-openai bundles vllm/distributed/.../lmcache_integration/ but the `lmcache`
# pip package (the actual store backend) is not preinstalled. Install it; --pre
# fallback in case only a pre-release wheel matches the container's vLLM.
LMCACHE_VERSION="${LMCACHE_VERSION:-}"
if [ -n "${LMCACHE_VERSION}" ]; then
    pip install --no-cache-dir "lmcache==${LMCACHE_VERSION}"
else
    pip install --no-cache-dir lmcache || pip install --no-cache-dir --pre lmcache
fi
python3 -c "import lmcache; print('[lmcache] installed', getattr(lmcache,'__version__','?'))" \
    || { echo "[lmcache] ERROR: import failed after install" >&2; exit 1; }

# save_only_first_rank fix: lmcache 0.4.5..0.4.7 assert lmcache_engine!=None in the
# request_finished abort branch, but non-first ranks have no engine -> AssertionError
# on request abort. Patch in an early-return guard (what the turbo "lmcachefix" image
# carries). Idempotent; non-fatal so a config not using save_only_first_rank still runs.
if [ -f /configs/patches/patch_lmcache_save_first_rank.py ]; then
    python3 /configs/patches/patch_lmcache_save_first_rank.py \
        || echo "[lmcache] WARN: save_only_first_rank patch failed (adapter anchors may differ)"
fi
