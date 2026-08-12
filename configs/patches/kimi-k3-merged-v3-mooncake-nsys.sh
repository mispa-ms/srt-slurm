#!/usr/bin/env bash
# k3-merged-v3 + Mooncake under DCP, with Nsight Systems installed.
#
# WHY THIS PAIR EXISTS. v4 is 6.0-10.2% below v3 Mooncake at c56 and c70 on a
# matched 8192 prefill budget, and two commit-level hypotheses have now been
# tested and refuted:
#   vllm#51726 (the >=160 GB _max_num_batched_tokens default moving 8192 ->
#     16384) -- holding v4 at 8192 made it worse, so that default was masking
#     part of the gap rather than causing it.
#   vllm#51739 (the long-context MLA gather rewrite, the only compiled change in
#     the 68 commits) -- reverting it left the gap unchanged (-10.2% against
#     -10.1% at c70), and its microbenchmark shows the new kernel is 1.65-1.76x
#     FASTER on this model's shapes, not slower.
#
# What survives both is the step cost: ITL p90 at c70 is 144.4 ms on v3 and
# 157.4 ms on v4 with GPU prefix hit, ISL and error count unchanged, and a
# repeat run put run-to-run spread at 2.0%. So the regression is real and it is
# in the step, but reading commit titles has now been wrong twice. These two
# arms stop guessing and measure which kernels moved.
#
# THE TWO ARMS MUST BE READ AS A PAIR. Same concurrency, same budget, same
# window placement, one image apart. A trace of either alone says nothing.
#
# -t cuda-sw,nvtx is set in the config, not here: on Blackwell the default
# "cuda" uses hardware-event tracing, which kills the worker's async output copy
# stream at capture teardown.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-merged-v3-mooncake.sh"

# vllm/vllm-openai ships no nsys, so the wrapped worker launch would exit 127.
bash "$(dirname "${BASH_SOURCE[0]}")/install-nsys-cli.sh"
