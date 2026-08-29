#!/usr/bin/env bash
# c96 no-spec baseline, plus all three cache PRs.
# =============================================================================
# c96 is where MI355X ATOM is strongest and our margin thinnest (+4.1% at their x 5.0), and where the measured Mooncake catch collapses to 8.8% against 58.2% at c32. No DSpark here, so #53614's spec-decode rewind cannot pay -- which makes this the control separating the spec path from the DCP geometry.
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

bash /configs/patches/vllm-container-deps-k3-b200-828c.sh
bash /configs/patches/vllm-container-deps-k3-pr51358.sh
bash /configs/patches/vllm-container-deps-k3-pr53614-828.sh
bash /configs/patches/vllm-container-deps-k3-pr53598.sh
