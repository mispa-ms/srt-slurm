#!/usr/bin/env bash
# The 08/28 DSpark + PP chain, plus the Nsight Systems CLI.
# =============================================================================
# WHY THIS EXISTS. Every nsys trace this workstream owns is a NO-SPEC arm --
# c48-lad-nospec, five of them. The frontier point is ns=4, and the two differ in
# the one quantity the kernel analysis turns on: tokens per decode step is
# 48 for no-spec (padded to the 53 capture size) and 48 x (4+1) = 240 for ns=4
# (padded to 256). cuBLAS picks split-K when the output does not fill the GPU, so
# a 5x change in M can flip that choice -- which means "split-K is 31% of the
# critical path" is currently an unverified claim about our best arm.
#
# The existing nsys config pairs with vllm-container-deps-k3-nsys.sh, which is
# hfshim + the CLI and carries none of the DSpark stack. This chain is the arm's
# own setup with the CLI appended, so the traced binary is the one that produced
# 9,553.7 tok/s/chip and not a neighbour of it.
# =============================================================================
set -euo pipefail

echo "=== k3-828mc-dspark-nsys: arm stack, then the nsys CLI ==="
bash /configs/patches/vllm-container-deps-k3-b200-828mc-dspark.sh

# The image already has the NVIDIA cuda repo registered — install directly.
# Do NOT add cuda-keyring on top: it triggers an apt Signed-By conflict.
#
# Package availability differs per distro/arch, so try newest-first and stop at
# the first hit. `nsight-systems-cli` is the fallback (older, but enough for
# --delay/--duration capture).
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    for pkg in nsight-systems-2025.6.3 nsight-systems-2025.3.1 nsight-systems nsight-systems-cli; do
        if apt-get install -y --no-install-recommends "$pkg"; then
            echo "[k3-nsys] installed $pkg"
            break
        fi
        echo "[k3-nsys] package $pkg unavailable, trying next"
    done
fi

# Fail loudly here rather than 127 lines later inside the worker launch.
command -v nsys >/dev/null 2>&1 || { echo "[k3-nsys] FATAL: nsys still not on PATH after install attempts" >&2; exit 1; }
nsys --version
echo "=== k3-nsys: done ==="
