#!/usr/bin/env bash
# Upstream #54167: the low-latency GEMM mixin never called its base initializer.
# =============================================================================
# The 2026-08-28 nightly (6f7df92a, built 06:13 UTC) carries #50572, which made
# UnquantizedLinearMethod resolve and store its GEMM implementation as _gemm_impl at
# init. Kimi-K3's low-latency mixin does not chain to that initializer, so any shape the
# fast path declines falls back into a method whose _gemm_impl was never set:
#
#     AttributeError: 'KimiK3LowLatencyLinearMethod' object has no attribute '_gemm_impl'
#
# #54167 is one line -- `super().__init__()` in _KimiK3LowLatencyApply.__init__ -- and it
# merged at 08:32 UTC, two hours after this image was built. So the fix exists upstream
# and is not in the image we would run.
#
# Reported by Hanjie Qiu against this nightly; we do not hit it on a9a17e7095 because
# that one predates #50572 and has no _gemm_impl at all.
# =============================================================================
set -euo pipefail

echo "=== pr54167: low-latency GEMM fallback initialization ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
LL="$VLLM_ROOT/vllm/models/kimi_k3/nvidia/low_latency_gemm.py"

if ! grep -q "_gemm_impl" "$VLLM_ROOT/vllm/model_executor/layers/linear.py"; then
    echo "[pr54167] this image predates #50572; there is no _gemm_impl to initialize"
elif grep -A2 "def __init__(self, plan: dict\[int, ResolvedCall\]) -> None:" "$LL" | grep -q "super().__init__()"; then
    echo "[pr54167] already present in this image"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-pr54167.patch > /tmp/pr54167-dry.log 2>&1; then
        echo "[pr54167] FATAL: does not apply to this image" >&2
        cat /tmp/pr54167-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-pr54167.patch
    echo "[pr54167] applied under $VLLM_ROOT"
fi

python3 - <<'PY'
import importlib.util
import os
import re
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/low_latency_gemm.py")).read()
lin = open(os.path.join(root, "vllm/model_executor/layers/linear.py")).read()

if "_gemm_impl" in lin:
    m = re.search(r"def __init__\(self, plan: dict\[int, ResolvedCall\]\) -> None:\n(.*?)\n\n", src, re.S)
    if not m or "super().__init__()" not in m.group(1):
        sys.exit("[pr54167] FATAL: the mixin still does not chain to its base initializer")
    print("[pr54167] verified: the mixin chains to its base initializer")
else:
    print("[pr54167] verified: nothing to do on this image")
PY

echo "=== pr54167: done ==="
