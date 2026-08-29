#!/usr/bin/env bash
# c48 ns=4 baseline, plus vLLM PR #51358 only.
# =============================================================================
# Prices #51358 (exact Mamba boundary states in the Mooncake store) alone. Merged upstream 2026-08-29T02:40Z, about 20h after this image was built, so the image does not carry it.
#
# The base chain is the one this arm's own baseline ran, so the only difference is the
# patch(es) named above. The three baselines do not share a chain -- c48 ns=4 ran
# 828mc-dspark (mambacache), c96 no-spec ran 828c and c24 ns=7 ran 828-dspark3 (both
# mambagroups2) -- and pricing against a different chain would fold two changes into one
# number.
#
# Our patches run last. They edit vllm/v1/core/* and the Mooncake store, and
# k3-mooncake-53324-828 edits four of the same files; the mamba* patches touch only
# mamba_hybrid.py and are disjoint from all three. Dry-run order, 0 rejects, 16 files
# compile: 6f7df92a -> 53324 -> 51358 -> 53614-828 -> 53598, all at fuzz 0.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828mc-dspark.sh
bash /configs/patches/vllm-container-deps-k3-pr51358.sh
