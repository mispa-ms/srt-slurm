#!/usr/bin/env bash
# The B200 DCP8 chain plus upstream PR #53798, the align-mode seed divisor.
# =============================================================================
# Six guards on consumers of a bad block-table column all failed. This is the
# producer: add_request seeds the align running-state block by dividing computed
# tokens by the scheduler's block size instead of the mamba group's. PR #53798's own
# regression test describes the outcome as silent corruption at moderate lengths and a
# CUDA illegal memory access past ~100k tokens -- and AgentX prompts reach 470k.
#
# See vllm-container-deps-k3-mambaseed.sh. Nothing else changes.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-mambaseed.sh
