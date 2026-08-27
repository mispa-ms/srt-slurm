#!/usr/bin/env bash
# The B200 DCP8 chain plus the workspace wsprobe.
# =============================================================================
# Seven index guards failed; the indices were always in range. The launch-blocking
# traceback puts the fault in a cudagraph replay's eager segment, which dereferences
# weak references captured earlier -- and the tensors involved come from the
# WorkspaceManager, which the V2 runner never locks. See
# vllm-container-deps-k3-wsprobe.sh.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-wsprobe.sh
