#!/usr/bin/env bash
# Kimi-K3 HF cache shim plus the Nsight Systems CLI.
# =============================================================================
# WHY: K3 runs need the hfshim (the 1.4 TB checkpoint is pre-staged, not pulled)
#   and nsys recipes need the nsys CLI, which vllm/vllm-openai does not ship.
#   srtctl prefixes the worker launch with `nsys profile ...`, so without the
#   CLI the launch exits 127 with "nsys: command not found".
#
# Time-based capture only (`profiling.type: nsys-time`). The window comes from
# --delay/--duration, so none of the cudaProfilerApi start/stop coordination
# and none of the CudaProfilerWrapper patching is involved.
#
# The nsys install below is lifted from vllm-container-deps-nsys.sh on
# misunp/nsys-time-setup (510b7dc2); kept as a copy rather than a source so this
# branch has no cross-branch dependency.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-hfshim.sh

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
