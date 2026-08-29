#!/usr/bin/env bash
# 08/28 no-spec, with the corrected mamba group-id fix.
# =============================================================================
# mambagroups (v1) forced the context's list everywhere and that was wrong: measured,
# PP0's real layout is [Mamba x4, MLA] and PP1's is [MLA, Mamba, Mamba], while the
# context claimed [0,1,2] on both. v2 derives the list from the config in hand and
# rebuilds a stale context instead. See vllm-container-deps-k3-mambagroups2.sh.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828.sh
bash /configs/patches/vllm-container-deps-k3-mambagroups2.sh
