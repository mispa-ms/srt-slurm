#!/usr/bin/env bash
# The nsys CLI install plus the trtllm-gen BatchedGemm cubin rollback.
# =============================================================================
# WHY both: the rollback arm needs a trace to answer which kernel actually ran,
#   and vllm/vllm-openai ships no nsys, so the worker launch would exit 127.
#   Neither script subsumes the other, so this composes the two.
#
# Order matters only in that the rollback must run after any pip step that could
# reinstall flashinfer; the nsys install is apt-only, so either order works.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-nsys.sh
bash /configs/patches/vllm-container-deps-fi-bmm-rollback.sh

echo "=== k3-nsys + fi-bmm-rollback: done ==="
