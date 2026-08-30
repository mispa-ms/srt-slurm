#!/usr/bin/env bash
# c48 ns=4 baseline, plus vLLM PR #54255 only.
# =============================================================================
# THE DECODE ROUND. Three cache patches were priced first and none paid: #53614 died at
# its own assertion on every arm, #51358 cost -9.0% at c48 ns=4 (it halved the Mooncake
# catch, 55.7% -> 26.8%) and crashed at c96, and #53598 came back -0.9%, inside the
# 2.4-9% run-to-run spread. EP was -2.1% at c96, closing that axis on B200 AGG too.
#
# They could not have paid. The latency split at c48 ns=4, per request, is TTFT p90
# 2.57 s against 974 output tokens x 38.2 ms = 37.15 s of decode: prefill is 6.5% of the
# time. Removing all of it is worth at most +6.9%, and the measured elasticity of
# throughput to prefix-cache miss is -0.234, not the -1 that y = P/(1-h)/N_gpu implies
# when P is wrongly held constant. Tokens are 99.3% input; time is 93.5% decode.
#
# These arms attack the 93.5%. At fixed concurrency y is proportional to 1/latency, so
# an ITL cut converts to throughput almost one for one.
#
# Prices #54255 alone: FlashInfer's packed fused KDA kernel, replacing three launches
# (short conv, recurrent KDA, gated norm) with one on pure speculative batches. The
# config forces kda_spec_decode_backend=flashinfer rather than leaving it on auto,
# because auto is fail-closed and would silently measure the baseline.
#
# Base chain is 828mc-dspark, the one the c48 ns=4 baseline ran (9,399.5 tok/s/chip
# @ x 25.73), so the only difference is the patch(es). Both PRs are pure Python, apply
# to 6f7df92a at fuzz 0, and touch only vllm/models/kimi_k3/nvidia/** plus
# vllm/utils/flashinfer.py -- disjoint from k3-dspark-pp-828, k3-mooncake-53324-828 and
# the mamba* patches.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828mc-dspark.sh
bash /configs/patches/vllm-container-deps-k3-pr54255.sh
