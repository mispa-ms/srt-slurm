#!/usr/bin/env bash
# The B200 DCP8 diag chain, plus the Nsight Systems CLI.
# =============================================================================
# For the MXFP4 nsys arm. Engine-identical to
# vllm-container-deps-k3-b200-dcp8-diag.sh, which is what the measured c48
# MXFP4 runs used -- the only difference is that nsys is on PATH, so srtctl's
# `nsys profile ...` worker prefix resolves.
#
# ORDER. Install first. It is an apt fetch that either works in a minute or does
# not work at all, and finding that out before the patch chain and a 1.4 TB
# checkpoint load is worth the reordering. Nothing in the chain depends on nsys
# being absent.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-nsys-install.sh
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
