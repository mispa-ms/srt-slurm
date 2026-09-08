#!/usr/bin/env bash
# Wei's GB300 stack on aws-pdx: same patch, cluster-local checkpoint.
# =============================================================================
# WHY A SEPARATE SCRIPT. bia is decommissioned 2026-09-14 and the B300 frontier
# has to be reproduced on aws-pdx. Nothing about the STACK changes -- this
# applies exactly the patch kimi-k3-wei-gb300.sh applies, byte for byte, so a
# number measured here is comparable to the bia frontier. What changes is where
# the 1.5 TB of weights live, and that is not a stack property.
#
# THE CHECKPOINT IS ALREADY ON THE CLUSTER, via JET. Rev 9f62e4e -- the one the
# bia frontier ran on -- is synced to aws-pdx-slurm-1_lustre at
#
#   /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/
#     model/moonshotai_kimi-k3/hf/hf-9f62e4e_orig      (96 shards, 1.5 TB)
#
# verified readable from our account. So there is no download: the hfshim
# materialises the standard HF cache layout under $HF_HOME with symlinks and
# the job opens a local path.
#
# DO NOT let the shim fall back to its bia default. Its built-in STAGED_DIR is
# /lustre/share/coreai_comparch_aarwlt/hf_repos/... which does not exist here;
# it would abort rather than download, which is the right failure, but the
# override is what makes it work. K3_STAGED_DIR is exported below and the shim
# refuses if the directory is missing, so a wrong path fails at setup rather
# than an hour into a run.
# =============================================================================
set -euo pipefail

: "${K3_STAGED_DIR:=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/moonshotai_kimi-k3/hf/hf-9f62e4e_orig}"
export K3_STAGED_DIR

echo "=== wei-gb300-pdx: staged checkpoint = $K3_STAGED_DIR ==="
if [[ ! -d "$K3_STAGED_DIR" ]]; then
    echo "wei-gb300-pdx: FATAL: $K3_STAGED_DIR is not a directory." >&2
    echo "  JET said rev 9f62e4e is synced on aws-pdx-slurm-1_lustre; re-check with" >&2
    echo "  jet_model.py, and note its 'path' subcommand picks the NEWEST artifact" >&2
    echo "  (f831ab66), which is synced nowhere -- query the 9f62e4e artifact by id." >&2
    exit 1
fi
n=$(ls "$K3_STAGED_DIR"/*.safetensors 2>/dev/null | wc -l)
echo "    shards visible: $n"
if [[ "$n" -lt 90 ]]; then
    echo "wei-gb300-pdx: FATAL: expected 96 safetensors shards, found $n." >&2
    exit 1
fi

# The HF cache shim first: the stack patch below imports vllm, and a missing
# checkpoint should fail here rather than after a 6,368-line patch.
bash /configs/patches/vllm-container-deps-k3-hfshim.sh

# Then the stack, unchanged.
bash /configs/patches/kimi-k3-wei-gb300.sh

echo "=== wei-gb300-pdx: done ==="
