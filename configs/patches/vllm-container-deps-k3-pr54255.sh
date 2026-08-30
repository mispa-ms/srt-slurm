#!/usr/bin/env bash
# vLLM PR #54255 -- FlashInfer's packed fused KDA kernel for K3 speculative decode.
# =============================================================================
# WHY THE DECODE SIDE. See vllm-container-deps-k3-pr54168.sh for the measurement that
# redirected this workstream: at c48 ns=4 the request latency is 2.57 s TTFT against
# 37.15 s of decode, so prefill is 6.5% of the time even though input is 99.3% of the
# tokens. The measured elasticity of throughput to prefix-cache miss is -0.234, not -1.
# Decode is the 93.5% surface.
#
# WHAT IT DOES. For eligible pure speculative-decode batches it replaces three separate
# launches -- short convolution, recurrent KDA, gated normalisation -- with one
# flashinfer.fused_kda_decode_packed call. Mixed and unsupported batches keep the
# existing vLLM path.
#
# WHY IT SHOULD APPLY TO US. Eligibility is SM100-family CUDA, supported Kimi-K3 shapes,
# and a pure speculative batch. We are B200 (SM100) running DSpark ns=4, and every point
# on our frontier above c8 is a speculative arm.
#
# WHY THE BACKEND IS FORCED RATHER THAN LEFT ON auto. The knob is
# additional_config.kda_spec_decode_backend and it defaults to "auto", which is
# fail-closed: if anything about the shapes or the FlashInfer build does not match, auto
# silently keeps the old path and the arm measures the baseline while looking like a
# result. This workstream has lost days to exactly that, so the config sets
# "flashinfer" explicitly and this script asserts the resolver exists. A hard failure at
# startup costs minutes; a silent no-op costs a four-hour arm and a wrong conclusion.
#
# Entirely Python (the kernel lives in FlashInfer, already in the image), applies to
# 6f7df92a at fuzz 0, and touches only kda.py and utils/flashinfer.py -- disjoint from
# k3-dspark-pp-828, which edits model.py and the spec_decode workers but not kda.py.
# =============================================================================
set -euo pipefail

echo "=== k3-pr54255: FlashInfer speculative KDA backend ==="

VLLM_ROOT=${VLLM_ROOT:-$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')}
command -v patch >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq patch; }

cd "$VLLM_ROOT"
KDA=vllm/models/kimi_k3/nvidia/kda.py

if grep -q "resolve_kda_spec_decode_backend" "$KDA" 2>/dev/null; then
  echo "[pr54255] already applied"
else
  patch -p1 --forward --dry-run --fuzz=0 < /configs/patches/k3-pr54255-828.patch
  patch -p1 --forward --fuzz=0 < /configs/patches/k3-pr54255-828.patch
  echo "[pr54255] applied"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
for rel in ("vllm/models/kimi_k3/nvidia/kda.py", "vllm/utils/flashinfer.py"):
    compile(open(os.path.join(root, rel)).read(), rel, "exec")

kda = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
if "resolve_kda_spec_decode_backend" not in kda:
    sys.exit("[pr54255] FATAL: resolve_kda_spec_decode_backend missing from kda.py")
if "kda_spec_decode_backend" not in kda:
    sys.exit("[pr54255] FATAL: the config knob is not read anywhere in kda.py")

import vllm.utils.flashinfer as fi

for f in ("has_flashinfer_fused_kda_decode_packed",
          "is_flashinfer_fused_kda_spec_decode_supported"):
    if not hasattr(fi, f):
        sys.exit(f"[pr54255] FATAL: {f} missing from vllm.utils.flashinfer")

# Say out loud whether the FlashInfer side is actually present in this image. If it is
# not, forcing the backend will fail at startup -- which is the intended outcome, but it
# should be legible here rather than 10 minutes later in a stack trace.
try:
    ok = fi.has_flashinfer_fused_kda_decode_packed()
except Exception as e:  # noqa: BLE001
    ok = f"probe raised {type(e).__name__}: {e}"
print(f"[pr54255] flashinfer fused_kda_decode_packed available: {ok}")

import vllm.models.kimi_k3.nvidia.kda  # noqa: F401

print("[pr54255] verified: both files compile, resolver + helpers present")
PY

echo "=== k3-pr54255: done ==="
