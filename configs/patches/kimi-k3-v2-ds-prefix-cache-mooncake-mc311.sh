#!/bin/bash
# Port of InferenceX PR #2444's kimi-k3-v2-ds-prefix-cache-mooncake.sh to bia.
#
# Unchanged from SA: the vLLM patch pair (#49291 / #50153 backport plus the
# hybrid-KV load-failure recovery) and the CUDA 13 Mooncake worker runtime.
#
# This replaces our Hanjie-package patches rather than stacking on them --
# patch_kimi_k3_v2_ds_prefix_cache.py touches the same mamba_hybrid.py and
# nixl/ code that remove-vllm-ds-layout-spec-assert.patch and
# vllm-nixl-ssm-spec-prefix-cache.patch do, so the two sets must not be mixed.
#
# The master is launched by srt-slurm from the recipe's mooncake_kv_store block
# (container kvcacheai/mooncake:0.3.12.post1-cuda13), not from this script, so
# nothing here starts a master.

set -euo pipefail

# VARIANT: pins the mooncake worker runtime to 0.3.11.post1 instead of 0.3.12.post1.
# 0.3.11 is the version our own recipe ran when load_get was last observed working
# on bia (1,152 ops / 324.9 GB at c1). Everything else matches the 0.3.12 script.
#
# Keep the Kimi-specific vLLM image and install the matching CUDA 13 Mooncake
# runtime into it. The generic Mooncake package ships a CUDA 12-linked master,
# while vllm/vllm-openai:kimi-k3 is CUDA 13-only.
bash /configs/patches/vllm-container-deps.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl

python3 -m pip install msgpack
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 || true
python3 -m pip install --no-deps 'mooncake-transfer-engine-cuda13==0.3.11.post1'

# Apply the same Kimi V2 DS prefix-cache and NIXL SSM-state fixes as the
# no-offload arm after dependency setup, so both arms run identical vLLM code.
python3 /configs/patches/patch_kimi_k3_v2_ds_prefix_cache.py

# The current Kimi image still assumes a single KV-cache group when an async
# external load loses a block to store eviction. Kimi has hybrid KDA/MLA groups,
# so backport the conservative full-prefix recompute fallback for that rare path.
python3 /configs/patches/patch_kimi_k3_mooncake_hma_recompute.py
