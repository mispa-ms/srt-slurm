#!/usr/bin/env bash
# The 2026-08-30 nightly (1dc464d426), with DSpark under PP and the interleave fix.
# =============================================================================
# Same chain as 828mc-dspark plus one patch. Verified file by file against the
# 1dc464d426 tree rather than assumed:
#
#   pr54167      skips  -- super().__init__() is already in low_latency_gemm.py
#   53324        skips  -- the "PCP/DCP > 1" refusal is gone from connector.py
#   int64idx     needed -- .to(tl.int64) absent; the IMA fix is still not upstream
#   dspark-pp    needed -- applies at fuzz 0
#   emptycache   needed
#   mambacache   needed -- the unkeyed `if self._mamba_spec is None:` guard is there
#   interleave   needed -- NEW on this image; see the script for why
#
# The interleave patch runs last because it edits config/vllm.py, which nothing else
# in the chain touches, and because it is the only one that is a no-op on 08/28 and
# 08/29 -- so this chain is safe to point at an older image if we ever need to.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-828mc-dspark.sh
bash /configs/patches/vllm-container-deps-k3-interleave.sh
