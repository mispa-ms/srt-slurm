#!/usr/bin/env bash
# The NVFP4 B200 chain, plus the Nsight Systems CLI.
# =============================================================================
# For the NVFP4 nsys arm. Engine-identical to
# vllm-container-deps-k3nvfp4-b200.sh, which is what the measured c48 NVFP4 run
# used -- including the JET-pinned checkpoint shim (f8c5234) and the NVFP4
# loader support. The only difference is that nsys is on PATH.
#
# ORDER. Install first, for the reason in the MXFP4 twin: an apt fetch that
# fails should fail before the checkpoint load, not after it.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-nsys-install.sh
bash /configs/patches/vllm-container-deps-k3nvfp4-b200.sh
