#!/usr/bin/env bash
# The Nsight Systems CLI, and nothing else.
# =============================================================================
# WHY THIS IS SPLIT OUT. vllm-container-deps-k3-nsys.sh already installs nsys,
# but it opens by chaining the MXFP4 hfshim. Every K3 recipe that wants a trace
# already runs its own hfshim inside its own chain -- the B200 DCP8 chain calls
# vllm-container-deps-k3-hfshim.sh, the NVFP4 chain calls
# vllm-container-deps-k3nvfp4-hfshim.sh -- so chaining that script would either
# run the wrong shim or run one twice. This file carries the install alone so a
# recipe can compose it with whatever chain it already has.
#
# vllm/vllm-openai ships no nsys. srtctl prefixes the worker launch with
# `nsys profile ...`, so without the CLI the launch exits 127 with
# "nsys: command not found" and the job fails with nothing else wrong.
#
# The install body is lifted verbatim from vllm-container-deps-k3-nsys.sh so the
# two cannot drift in how they resolve a package.
# =============================================================================
set -euo pipefail

# The image already has the NVIDIA cuda repo registered -- install directly.
# Do NOT add cuda-keyring on top: it triggers an apt Signed-By conflict.
#
# Package availability differs per distro/arch, so try newest-first and stop at
# the first hit. `nsight-systems-cli` is the fallback (older, but enough for
# --delay/--duration capture).
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    for pkg in nsight-systems-2025.6.3 nsight-systems-2025.3.1 nsight-systems nsight-systems-cli; do
        if apt-get install -y --no-install-recommends "$pkg"; then
            echo "[nsys-install] installed $pkg"
            break
        fi
        echo "[nsys-install] package $pkg unavailable, trying next"
    done
fi

# Fail loudly here rather than 127 lines later inside the worker launch.
command -v nsys >/dev/null 2>&1 || {
    echo "[nsys-install] FATAL: nsys still not on PATH after install attempts" >&2
    exit 1
}
nsys --version
echo "=== nsys-install: done ==="
