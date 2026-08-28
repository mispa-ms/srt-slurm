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

exec bash /configs/apply-vllm-k3-nightly-aug28-baseline.sh
