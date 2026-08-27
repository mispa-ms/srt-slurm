#!/usr/bin/env bash
# Kimi-K3 GB300 PP arms + online FP8 on the layers the checkpoint left in BF16.
# =============================================================================
# WHY. moonshotai/Kimi-K3 is a compressed-tensors mxfp4-pack-quantized
# checkpoint whose own ignore list is
#
#   re:.*self_attn.*  re:.*shared_experts.*  re:.*mlp\.(gate|up|gate_up|down)_proj.*
#   re:.*lm_head.*    re:.*vision_tower.*    re:.*mm_projector.*
#
# so MXFP4 covers the routed experts and nothing else: attention projections,
# shared experts and the dense MLP all run BF16. AMD's ATOM submission casts
# exactly those to per-token-per-channel FP8 at load time
# (--online_quant_config ptpc_fp8, same exclude set minus the KDA conv1d), which
# is free speed we are not taking.
#
# vLLM could not do this: one quant config per model, and a checkpoint that
# declares a quant_method makes --quantization-config an error. vllm#51392
# composes the two -- the checkpoint method keeps its pre-quantized layers, the
# online method gets the rest. This script carries it.
#
# WHAT THIS IS NOT. It does not turn anything on by itself. The arm still has to
# pass --quantization-config; without it this script is the base stack plus
# dead code, which is what makes the A/B single-variable.
#
# BASE. Chains kimi-k3-gb300-pp.sh unchanged. The two patch sets share no file
# (k3-engine-0819 touches the connectors, scheduler, spec-decode and KDA
# metadata; 51392 touches the quantization layer and mla.py), so order is free
# and a conflict here would mean one of them moved.
# =============================================================================
set -euo pipefail

bash /configs/patches/kimi-k3-gb300-pp.sh

echo "=== k3-ptpc: apply vllm#51392 (online quant over a partial checkpoint) ==="

PATCH=/configs/patches/vllm-pr51392-online-partial-quant.patch
[ -f "${PATCH}" ] || PATCH=/configs/vllm-pr51392-online-partial-quant.patch
if [ ! -f "${PATCH}" ]; then
    echo "[k3-ptpc] FATAL: patch not found at /configs/patches/ or /configs/" >&2
    exit 1
fi

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[k3-ptpc] vllm root: ${VLLM_ROOT}"

if grep -q "get_effective_quant_method" \
        "${VLLM_ROOT}/vllm/model_executor/layers/quantization/base_config.py"; then
    echo "[k3-ptpc] already applied"
else
    if ! command -v patch >/dev/null 2>&1; then
        echo "[k3-ptpc] installing patch(1)"
        apt-get update -qq && apt-get install -y -qq patch
    fi
    # Dry-run first, for the reason the base script gives: a half-applied
    # quantization layer serves happily and is wrong in a way throughput does
    # not show.
    if ! patch -p1 -d "${VLLM_ROOT}" --batch --dry-run --forward < "${PATCH}" > /tmp/k3-ptpc-dry.log 2>&1; then
        echo "[k3-ptpc] FATAL: patch does not apply to this image." >&2
        echo "[k3-ptpc] Generated against nightly 5a4c8d9924 from vllm-project/vllm#51392;" \
             "if the container tag moved, regenerate it from that PR." >&2
        cat /tmp/k3-ptpc-dry.log >&2
        exit 1
    fi
    patch -p1 -d "${VLLM_ROOT}" --batch --forward < "${PATCH}"
    echo "[k3-ptpc] applied"
fi

# Assert by value. Each check names a specific thing the arm depends on, because
# the failure mode of a partial apply is a server that starts and quantizes
# nothing -- indistinguishable from the control at a glance, and a wasted pair.
python3 - <<'PY'
import importlib.util, os, sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
def src(p):
    return open(os.path.join(root, p)).read()

fail = []

base_config = src("vllm/model_executor/layers/quantization/base_config.py")
if "def get_effective_quant_method" not in base_config:
    fail.append("the composed resolver is missing from QuantizationConfig")
if "online_quantization_config" not in base_config:
    fail.append("QuantizationConfig carries no online_quantization_config slot")

# The composition is opt-in through --quantization-config, and on main that flag
# is rejected outright whenever the checkpoint declares a quant_method. If this
# refusal survives, the arm dies at startup instead of running unquantized.
if "quantization_config is only supported when quantization is" in src(
        "vllm/config/quantization.py"):
    fail.append("the checkpoint/online mutual-exclusion refusal is still present")

if "maybe_compose_online_quantization" not in src(
        "vllm/model_executor/model_loader/weight_utils.py"):
    fail.append("get_quant_config still returns the checkpoint config uncomposed")

# The call sites. linear.py covers every projection K3 builds; mla.py is the
# K3-specific one the PR converts by hand. If either still calls the plain
# resolver those layers stay BF16 and the arm silently measures the control.
if "get_effective_quant_method" not in src("vllm/model_executor/layers/linear.py"):
    fail.append("LinearBase still resolves through get_quant_method")
if "get_effective_quant_method" not in src("vllm/models/kimi_k3/nvidia/mla.py"):
    fail.append("K3 MLA still resolves through get_quant_method")

# The method the arm actually asks for. fp8_per_channel is per-output-channel
# weight scale + dynamic per-token activation -- the same recipe ATOM spells
# ptpc_fp8 -- and it needs a kernel that honours per-token activation quant.
if "Fp8PtpcOnlineLinearMethod" not in src(
        "vllm/model_executor/layers/quantization/online/fp8.py"):
    fail.append("the PTPC online linear method is missing")
if "fp8_per_channel" not in src("vllm/config/quantization.py"):
    fail.append("the fp8_per_channel shorthand is missing")

if fail:
    sys.exit("[k3-ptpc] FATAL:\n  - " + "\n  - ".join(fail))
print("[k3-ptpc] verified: composed resolver in, refusal gone, "
      "LinearBase + K3 MLA converted, PTPC method present")
PY

echo "=== k3-ptpc: done ==="
