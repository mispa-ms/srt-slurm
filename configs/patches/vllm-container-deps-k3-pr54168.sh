#!/usr/bin/env bash
# vLLM PR #54168 -- optimise the SM100 Kimi-K3 low-M fused latent-MoE decode tail.
# =============================================================================
# WHY THE DECODE SIDE, AFTER THREE CACHE PATCHES CAME BACK EMPTY. The earlier round was
# aimed at the prefix-cache miss on the argument that the AgentX trace is 99.3% input
# tokens, so y = P/(1-h)/N_gpu makes h multiplicative. The equation is true by
# definition but it is not a lever, because P is not constant. Measured directly: the
# #51358 arm moved the true miss 6.68% -> 9.98% (1.49x) and throughput only 9,399.5 ->
# 8,556.4 (0.91x), an elasticity of -0.234 rather than -1.
#
# The reason is in the latency split at c48 ns=4, per request:
#
#     output tokens          974
#     TTFT p90              2.57 s
#     decode = 974 x ITL   37.15 s   (ITL p90 38.2 ms)
#     TTFT share             6.5%
#
# At fixed concurrency y is proportional to 1/request-latency, and prefill work only
# lives inside TTFT. Removing *all* of it is worth at most +6.9%. Tokens are 99.3%
# input; time is 93.5% decode. Those are different fractions and the first one is not
# the one that pays.
#
# So this arm and its sibling (#54255) attack the 93.5%.
#
# WHAT IT DOES. Reduces SM100 latent-MoE decode tail latency at low token counts:
# deferred BF16 handling, a specialised path for M <= 16, wide vector loads of the 16
# route indices and BF16 route weights, dropped/invalid routes predicated before their
# GEMM2 fragments are gathered, and accumulation straight into FP32 registers with
# fma.rn.f32.bf16. Merged upstream 2026-08-28T17:45Z; our image was built 06:13Z the
# same day, about eleven hours earlier, so it is not in it.
#
# WHY IT CAN BE PATCHED AT ALL. It is entirely Python -- the kernels are CuTe DSL, JIT
# compiled at run time, so there is nothing to build. Its sibling #54261 (native CUDA
# AttnRes as the SM100 default) touches attn_res_kernel.cu and torch_bindings.cpp and
# therefore cannot be applied this way; it needs an image build.
#
# WE ALREADY RUN THIS PATH. Every arm sets VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1, so
# the fused tail is live and this changes its cost rather than switching it on.
#
# Applies to 6f7df92a at fuzz 0, and touches only
# vllm/models/kimi_k3/nvidia/ops/**, disjoint from every other patch in the chain.
# =============================================================================
set -euo pipefail

echo "=== k3-pr54168: low-M fused latent MoE tail ==="

VLLM_ROOT=${VLLM_ROOT:-$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')}
command -v patch >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq patch; }

cd "$VLLM_ROOT"
PRIM=vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py

if grep -q "_SEVEN_CTA_MAX_M" "$PRIM" 2>/dev/null; then
  echo "[pr54168] already applied"
else
  patch -p1 --forward --dry-run --fuzz=0 < /configs/patches/k3-pr54168-828.patch
  patch -p1 --forward --fuzz=0 < /configs/patches/k3-pr54168-828.patch
  echo "[pr54168] applied"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
base = "vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail"
FILES = (
    f"{base}/allreduce_rmsnorm_reduce_scatter_early_exit.py",
    f"{base}/fused_add_multicast_skinny_gemm.py",
    f"{base}/lamport_copy.py",
    f"{base}/primitives.py",
    "vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py",
)
for rel in FILES:
    compile(open(os.path.join(root, rel)).read(), rel, "exec")

prim = open(os.path.join(root, f"{base}/primitives.py")).read()
for s in ("_SEVEN_CTA_MAX_M", "_LAMPORT_COPY_THREADS"):
    if s not in prim:
        sys.exit(f"[pr54168] FATAL: {s} missing from primitives.py")

# The whole point is the low-M specialisation. If the module imports but the fused tail
# is never enabled, this arm measures the baseline and says nothing -- so state the
# coupling here where it is cheap to notice.
if os.environ.get("VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION") not in ("1", "true", "True"):
    print(
        "[pr54168] WARNING: VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION is not set to 1; "
        "the fused tail this patch optimises may not run at all"
    )

import vllm.models.kimi_k3.nvidia.ops.latent_moe_tail  # noqa: F401

print("[pr54168] verified: 5 files compile, low-M symbols present, module imports")
PY

echo "=== k3-pr54168: done ==="
