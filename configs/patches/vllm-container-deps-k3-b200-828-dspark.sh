#!/usr/bin/env bash
# The 2026-08-28 nightly (6f7df92a), with DSpark under PP.
# =============================================================================
# The no-spec chain plus the two things speculation needs on this image:
#
#   dspark-pp-828   lifts the PP+spec refusal. Re-ported for 6f7df92a: the a9a17e7095
#                   version needed one more hunk because 08/28 added fused_qkv_a_g_proj
#                   to packed_modules_mapping, displacing the class attribute we insert.
#   emptycache      returns the loader's cached blocks to the driver before the draft
#                   model builds its own direct-DCP symmetric-memory workspace
#
# emptycache runs last on purpose: it anchors on the single line calling the speculator
# loader, and that line survives dspark-pp-828 unchanged (verified: exactly one match
# before and after).
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pp-828.sh
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh
