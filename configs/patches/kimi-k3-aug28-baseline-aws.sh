#!/usr/bin/env bash
# Hanjie's Aug 28 Kimi-K3 baseline stack, with the two things aws-cmh needs in
# front of it.
#
# The stack itself is untouched: apply-vllm-k3-nightly-aug28-baseline.sh on
# nightly 6f7df92a8, which carries vLLM PR #53324 (MooncakeStore with hybrid DCP
# prefix caching -- still open upstream, so no nightly has it), the k3-agent-all
# supplemental, the two K3 env gates, and the 64-bit prefill checkpoint index.
# PR #53682 and #53773 are ancestors of that nightly and are deliberately not
# replayed. Online quantization (#51392) and its meta-device companion are not
# here; that is what the -online-quant- script is for.
#
# What this file adds, and only this:
#   1. K3_STAGED_DIR. srtctl resolves moonshotai/Kimi-K3 through a model_paths
#      alias on Hanjie's cluster; aws-cmh has none, and without a staged path
#      the job downloads 1.45 TB inside a 4-hour allocation.
#   2. the HF shim, which makes that staged directory usable in place of a
#      download.
set -euo pipefail

if [ -z "${K3_STAGED_DIR:-}" ]; then
    for _cand in \
        /scratch/fsw/portfolios/coreai/users/kdhruv/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3 \
        /lustre/share/coreai_comparch_inferencex/models/kimi-k3
    do
        if [ -d "${_cand}" ]; then
            export K3_STAGED_DIR="${_cand}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3-aug28] FATAL: no staged checkpoint. Set K3_STAGED_DIR." >&2
    exit 1
fi
echo "[k3-aug28] staged checkpoint: ${K3_STAGED_DIR}"

bash /configs/patches/vllm-container-deps-k3-hfshim.sh

bash /configs/apply-vllm-k3-nightly-aug28-baseline.sh

# One more, and it is upstream's rather than this stack's: vllm#54167.
#
# enable_kimi_k3_low_latency_gemm swaps a fresh KimiK3LowLatencyLinearMethod
# onto every unquantized K3 linear whose (N, K) is in the measured table. That
# class inherits _KimiK3LowLatencyApply before UnquantizedLinearMethod, and the
# mixin's __init__ never chains, so UnquantizedLinearMethod.__init__ -- the only
# place _gemm_impl is assigned -- does not run. apply() falls back to
# super().apply() the moment the plan has no entry for the current token count,
# which the 16384-token profile_run guarantees:
#
#   AttributeError: 'KimiK3LowLatencyLinearMethod' object has no attribute
#   '_gemm_impl'   (kda.py:608 in_proj_qkvgfab -> linear.py:589)
#
# This does not fire on the online-quant arms: with linear=fp8_per_channel the
# affected layers are no longer UnquantizedLinearMethod, so the swap skips them
# and the missing __init__ is never reached. Dropping online quantization is
# what exposes it. Upstream merged the same one-line change at 08:32Z on
# 2026-08-28, two hours after this nightly was cut.
VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
LL_PATCH=/configs/patches/vllm-k3-lowlatency-linear-init-on-6f7df92a8.patch
if patch --batch --forward --dry-run -d "${VLLM_ROOT}" -p1 < "${LL_PATCH}" >/dev/null; then
    patch --batch --forward -d "${VLLM_ROOT}" -p1 < "${LL_PATCH}"
    echo "[k3-aug28] applied low-latency linear __init__ chain"
elif patch --batch --reverse --dry-run -d "${VLLM_ROOT}" -p1 < "${LL_PATCH}" >/dev/null; then
    echo "[k3-aug28] low-latency linear __init__ chain already present"
else
    echo "[k3-aug28] FATAL: low-latency linear patch does not apply." >&2
    exit 1
fi

python3 -c 'import importlib.util,os;r=os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin));s=open(r+"/vllm/models/kimi_k3/nvidia/low_latency_gemm.py").read();assert "super().__init__()\n        self._plan" in s, "low-latency __init__ chain missing";print("[k3-aug28] verified: _gemm_impl will be set on the swapped-in method")'

