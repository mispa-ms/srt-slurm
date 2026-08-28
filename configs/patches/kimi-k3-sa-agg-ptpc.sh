#!/usr/bin/env bash
# SA's submitted Kimi-K3 GB300 AGG arm, on oci-aga, plus online FP8 and nothing
# else.
# =============================================================================
# WHAT THIS IS. The A/B that prices dense FP8 has to run on the configuration
# whose curve we are trying to move, not on ours. SA's submitted arm is
#   InferenceX benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic/
#   agg-gb300-dcp8-nospec-mooncake-agentic.yaml
# on the stock nightly vllm/vllm-openai:nightly-dev-arm64-cu13.0.1-75c2eef.
# It carries no patch of ours, so neither does this script.
#
# WHAT IT APPLIES, in full:
#   1. resolve the staged K3 checkpoint (oci-aga has no /scratch/models alias)
#   2. vllm#51392, so online quantization can compose with the checkpoint's MXFP4
#   3. one line of ours, so the DCP workspace is not allocated on meta
#
# WHAT IT DELIBERATELY DOES NOT APPLY. k3-engine-0819.patch, which our own arms
# carry. SA runs the stock nightly and it is a different nightly -- 75c2eef
# (08-14) against the 5a4c8d99 (08-19) that patch is generated on. Carrying it
# would make this a third configuration rather than SA's.
#
# THE SHIM. quantization/utils/config_utils.py does not exist at 75c2eef; it
# arrives upstream later. 51392 adds one function to it, used only to group a
# log line, so the module is created with that function and nothing else.
# =============================================================================
set -euo pipefail

if [ -z "${K3_STAGED_DIR:-}" ]; then
    for _cand in \
        /lustre/share/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/hanjieq/models/kimi-k3 \
        /lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3
    do
        if [ -d "${_cand}" ]; then
            export K3_STAGED_DIR="${_cand}"
            echo "[k3-sa] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3-sa] FATAL: no staged checkpoint on this cluster. Set K3_STAGED_DIR." >&2
    echo "[k3-sa] Refusing to continue -- the fallback is a 1.45 TB download inside" \
         "a 4-hour job, which fails later and less legibly than this does." >&2
    exit 1
fi

bash /configs/patches/vllm-container-deps-k3-hfshim.sh

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[k3-sa] vllm root: ${VLLM_ROOT}"

CFGUTILS="${VLLM_ROOT}/vllm/model_executor/layers/quantization/utils/config_utils.py"
if [ ! -f "${CFGUTILS}" ]; then
    echo "[k3-sa] creating the config_utils shim (absent at this base)"
    cat > "${CFGUTILS}" <<'SHIM'
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Backport shim: this base predates the module upstream carries.
vllm#51392 needs only get_layer_name_after_index, for its log summary."""


def get_layer_name_after_index(layer_name: str) -> str:
    """Return the suffix following the final numeric component of a layer name."""
    parts = layer_name.split(".")
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].isdigit():
            return ".".join(parts[index + 1 :])
    return layer_name
SHIM
fi

apply_patch() {
    local label="$1" sentinel="$2" file="$3" patch="$4"
    [ -f "${patch}" ] || patch="/configs/$(basename "${patch}")"
    if [ ! -f "${patch}" ]; then
        echo "[k3-sa] FATAL: ${label} patch not found" >&2
        exit 1
    fi
    if grep -q "${sentinel}" "${VLLM_ROOT}/${file}"; then
        echo "[k3-sa] ${label} already applied"
        return
    fi
    if ! command -v patch >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq patch
    fi
    # Dry-run first. A half-applied quantization layer serves happily and is
    # wrong in a way throughput does not show.
    if ! patch -p1 -d "${VLLM_ROOT}" --batch --dry-run --forward < "${patch}" > "/tmp/k3-sa-${label}.log" 2>&1; then
        echo "[k3-sa] FATAL: ${label} does not apply to this image." >&2
        cat "/tmp/k3-sa-${label}.log" >&2
        exit 1
    fi
    patch -p1 -d "${VLLM_ROOT}" --batch --forward < "${patch}"
    echo "[k3-sa] ${label} applied"
}

echo "=== k3-sa: apply vllm#51392 (online quant over a partial checkpoint) ==="
apply_patch pr51392 get_effective_quant_method \
    vllm/model_executor/layers/quantization/base_config.py \
    /configs/patches/vllm-pr51392-online-partial-quant-75c2eef.patch

echo "=== k3-sa: apply the K3 DCP meta-device fix ==="
apply_patch dcpmeta kv_b_proj_device \
    vllm/models/kimi_k3/nvidia/mla.py \
    /configs/patches/vllm-k3-dcp-device-under-meta.patch

# Assert by value. A partial apply produces a server that starts and quantizes
# nothing, which is indistinguishable from the control at a glance.
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
if "quantization_config is only supported when quantization is" in src(
        "vllm/config/quantization.py"):
    fail.append("the checkpoint/online mutual-exclusion refusal is still present")
if "maybe_compose_online_quantization" not in src(
        "vllm/model_executor/model_loader/weight_utils.py"):
    fail.append("get_quant_config still returns the checkpoint config uncomposed")
if "get_effective_quant_method" not in src("vllm/model_executor/layers/linear.py"):
    fail.append("LinearBase still resolves through get_quant_method")

mla = src("vllm/models/kimi_k3/nvidia/mla.py")
if "get_effective_quant_method" not in mla:
    fail.append("K3 MLA still resolves through get_quant_method")
# Without this the DCP query-gather workspace takes its device from kv_b_proj,
# which online quant stages on meta, and symmetric memory cannot back meta.
if "kv_b_proj_device" not in mla:
    fail.append("K3 MLA still reads its DCP device off kv_b_proj unguarded")

if "Fp8PtpcOnlineLinearMethod" not in src(
        "vllm/model_executor/layers/quantization/online/fp8.py"):
    fail.append("the PTPC online linear method is missing")
if "fp8_per_channel" not in src("vllm/config/quantization.py"):
    fail.append("the fp8_per_channel shorthand is missing")

# This arm is SA's, so our engine patch must NOT be here. If it is, the run is a
# third configuration and its delta cannot be read against their curve.
if "The drafter is instantiated only on the last pipeline stage" in src(
        "vllm/config/speculative.py"):
    fail.append("k3-engine-0819 is present; this is no longer SA's stack")

if fail:
    sys.exit("[k3-sa] FATAL:\n  - " + "\n  - ".join(fail))

import vllm
print(f"[k3-sa] vllm {getattr(vllm, '__version__', '?')}")
print("[k3-sa] verified: 51392 composed, DCP meta guard in, engine patch absent")
PY

echo "=== k3-sa: done ==="
