#!/usr/bin/env bash
# Wei Zhao's balanced-router diagnostic, applied to the pinned image.
# =============================================================================
# WHAT IT DOES. Adds VLLM_MOE_BALANCED_ROUTER_LOGITS. When set, moe_runner.py
# replaces the router logits with a deterministic round-robin pattern so every
# expert receives floor or ceil of the assignments -- maximally balanced. The MoE
# backend is untouched; only who each token routes to changes.
#
#   wzhao18/vllm 19a843c002c3b6508d4735b1454c805cccb66ace
#   "[MoE] Add deterministic balanced routing diagnostic"
#
# WHY. Wei measured mxfp4 MoE well ahead of nvfp4 on real (imbalanced) routing and
# level with it once balance is forced, and explained it: "imbalance makes the
# performance better as fewer experts need to be loaded". Our AgentX arms run real
# traffic, so they sit in the imbalanced case -- MXFP4's best. This makes the
# opposite case measurable.
#
# THE OUTPUT IS GARBAGE, ON PURPOSE. Routing is fake, so generated tokens are
# meaningless and the run carries no accuracy signal. It also means output lengths
# move, because EOS now fires wherever the fake routing sends it, so a
# balanced-vs-original comparison inside one precision is not clean. What survives
# is the cross-precision ratio: NVFP4/MXFP4 decode cost per generation token, taken
# under balance and compared against the same ratio taken without it (1.270 on the
# -iterlog pair). Both arms carry the same distortion, so that ratio is readable.
#
# Verified to apply to 728d3ad with --fuzz=0, all hunks, offsets only. The K3 MoE
# reaches the patched call site: model.py:624 builds its experts through
# FusedMoEFactory, and the hook sits in _apply_quant_method just before
# forward_monolithic. It is on the moe-backend auto path, which is the path every
# number on this workstream was measured on.
# =============================================================================
set -euo pipefail

VLLM_ROOT=${VLLM_ROOT:-$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')}
command -v patch >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq patch; }

if grep -q "VLLM_MOE_BALANCED_ROUTER_LOGITS" "$VLLM_ROOT/vllm/envs.py"; then
    echo "[balrouter] already present; skipping"
else
    echo "=== balrouter: applying the balanced-router diagnostic ==="
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
            < /configs/patches/k3-moe-balanced-router.patch > /tmp/balrouter-dry.log 2>&1; then
        echo "[balrouter] FATAL - patch does not apply to this image" >&2
        cat /tmp/balrouter-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-moe-balanced-router.patch
fi

# Verify by the symbols the run actually needs, not by "patch returned 0".
python3 - <<'PY'
import importlib, sys
import vllm.envs as envs
assert hasattr(envs, "VLLM_MOE_BALANCED_ROUTER_LOGITS"), "env var missing after patch"
m = importlib.import_module("vllm.model_executor.layers.fused_moe.runner.moe_runner")
assert hasattr(m, "balanced_router_logits"), "balanced_router_logits missing after patch"
print("[balrouter] verified: env var + balanced_router_logits present, flag=%s"
      % envs.VLLM_MOE_BALANCED_ROUTER_LOGITS)
PY
echo "=== balrouter: done ==="
