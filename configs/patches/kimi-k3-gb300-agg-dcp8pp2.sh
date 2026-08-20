#!/usr/bin/env bash
# Kimi-K3 GB300 AGG, TP8 x DCP8 x PP2, on a stock vLLM nightly.
# =============================================================================
# WHAT THE STACK IS. One patch, k3-all-on-0819.patch: every non-upstream commit
# of xinli-sw/vllm@k3-agent-all, rebased onto the 2026-08-19 nightly
# (5a4c8d99242e9e069b604d0e9b969e77f7dd501d). 15 files, 43 hunks. Ten commits:
#
#   Kimi-K3 DCP: hybrid CPU-offload block accounting        (block-table width
#                                                            guard + offload)
#   Kimi-K3: hybrid-aware KV-load-failure recompute
#   DSpark under DCP on AGG v2
#   Mooncake store: support DCP with hybrid attention
#   Carry vllm#50359 + two guards on its retry
#   Fill DCP local seq lens on dummy input batches          (Xin Li)
#   Adopt upstream PR #50514: spec decode under PP          + two loader fixes
#
# An eleventh, our carry of vllm#50493, is deliberately absent: it merged
# upstream on 2026-08-18 and rebased to an empty commit.
#
# WHY ONE PATCH AND NOT A CHAIN. Assembling this from the individual
# k3-dcp-*.patch files does not work on this base -- four of six fail, because
# upstream moved under them and in two places fixed the same defect better:
#   - the Mamba block table is no longer DCP-scaled at the call site; upstream
#     delegates to spec.max_num_blocks_per_req(), which is the same rule stated
#     in the spec instead of at each caller.
#   - the DFlash speculator already computes DCP-local slots via cp_local_slot
#     and adds null-block guarding (ctx_block_id != 0) on top.
# Taking the rebased branch keeps those upstream improvements instead of
# reverting them, which a patch chain would have done silently.
#
# WHAT IS NOT HERE. No Wei agentx-v2 patch and no nightly-tag pin. Everything
# K3 needs is upstream now: the model tree, TOKENSPEED_MLA, FLASHINFER_MLA,
# deep_gemm_mega_moe, the DSpark speculator, and -- since #50484 (08-10) and
# #50493 (08-18) -- K3 DCP itself. The aug17 v2 patch also would not apply: it
# fails 5 hunks across 4 files on this nightly.
#
# WHY AN ACCURACY ARM SHIPS WITH EVERY PERF ARM. The perf harness runs
# ignore_eos and carries no accuracy signal, so it reports a fast, wrong server
# as a fast one. The PP adopting commit's own account of what five earlier
# attempts missed: "draft tokens did not reach the earlier stages, so the first
# rank embedded PLACEHOLDER_TOKEN_ID(-1)" -- silent corruption. Under DCP the
# same applies to KV mapping. Treat any throughput number from these arms as
# unconfirmed until its GSM8K twin lands.
# =============================================================================
set -euo pipefail

# The staged checkpoint sits at a different path on every cluster. An explicit
# K3_STAGED_DIR wins; otherwise take the first candidate that exists and say
# which one, because guessing costs a whole run to find out.
if [ -z "${K3_STAGED_DIR:-}" ]; then
    for _cand in \
        /lustre/share/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/hanjieq/models/kimi-k3 \
        /lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3
    do
        if [ -d "${_cand}" ]; then
            export K3_STAGED_DIR="${_cand}"
            echo "[k3-dcp8pp2] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3-dcp8pp2] FATAL: no staged checkpoint on this cluster. Set K3_STAGED_DIR." >&2
    echo "[k3-dcp8pp2] Refusing to continue -- the fallback is a 1.45 TB download inside" \
         "a 4-hour job, which fails later and less legibly than this does." >&2
    exit 1
fi

bash /configs/patches/vllm-container-deps-k3-hfshim.sh

echo "=== k3-dcp8pp2: apply k3-all-on-0819 ==="

PATCH=/configs/patches/k3-all-on-0819.patch
[ -f "${PATCH}" ] || PATCH=/configs/k3-all-on-0819.patch
if [ ! -f "${PATCH}" ]; then
    echo "[k3-dcp8pp2] FATAL: patch not found at /configs/patches/ or /configs/" >&2
    exit 1
fi

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[k3-dcp8pp2] vllm root: ${VLLM_ROOT}"

if grep -q "The drafter is instantiated only on the last pipeline stage" \
        "${VLLM_ROOT}/vllm/config/speculative.py"; then
    echo "[k3-dcp8pp2] already applied"
else
    if ! command -v patch >/dev/null 2>&1; then
        echo "[k3-dcp8pp2] installing patch(1)"
        apt-get update -qq && apt-get install -y -qq patch
    fi
    # Dry-run first. A partially applied engine is worse than a failed job: it
    # starts, serves, and is wrong in a way no perf metric shows.
    if ! patch -p1 -d "${VLLM_ROOT}" --batch --dry-run --forward < "${PATCH}" > /tmp/k3-all-dry.log 2>&1; then
        echo "[k3-dcp8pp2] FATAL: patch does not apply to this image." >&2
        echo "[k3-dcp8pp2] The patch is generated against nightly 5a4c8d9924; if the" \
             "container tag moved, regenerate it from mispa-ms/vllm@misunp/k3-all-on-0819." >&2
        cat /tmp/k3-all-dry.log >&2
        exit 1
    fi
    # --batch: without it, a patch naming a file the image does not have
    # prompts "File to patch:" and reads the answer from stdin -- which is
    # the patch itself. It then eats patch lines as answers and applies a
    # truncated diff while reporting success.
    patch -p1 -d "${VLLM_ROOT}" --batch --forward < "${PATCH}"
    echo "[k3-dcp8pp2] applied"
fi

# Assert by value, not by presence. A verifier that asks hasattr of an attribute
# set in __init__ is always False on the class and killed three arms at startup
# before anyone noticed it could not pass.
python3 - <<'PY'
import importlib.util, os, sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
def src(p):
    return open(os.path.join(root, p)).read()

fail = []

if "DSpark does not support pipeline parallelism" in src(
        "vllm/v1/worker/gpu/spec_decode/dspark/utils.py"):
    fail.append("the upstream DSpark-under-PP refusal is still present")
if "pipeline_parallel_size=1" not in src("vllm/config/speculative.py"):
    fail.append("draft ParallelConfig still inherits the target PP size")
if "supports_aux_hidden_states_over_pp" not in src("vllm/models/kimi_k3/nvidia/model.py"):
    fail.append("K3 does not opt in to relaying aux hidden states across PP stages")
if "_check_block_table_width" not in src("vllm/models/kimi_k3/nvidia/kda_metadata.py"):
    fail.append("KDA block-table width guard missing")

# These two are upstream's, not ours, and this arm depends on both. #50484 gave
# K3 DCP; #50493 made partial prefix-cache hits work under it, without which
# prefix matching silently drops to the DCP-scaled group block.
if "MLADCPManager" not in src("vllm/models/kimi_k3/nvidia/mla.py"):
    fail.append("K3 DCP (#50484) missing from this image")
if "dcp_world_size > 1" not in src("vllm/v1/core/kv_cache_coordinator.py"):
    fail.append("DCP-aware partial hash hits (#50493) missing from this image")

if fail:
    sys.exit("[k3-dcp8pp2] FATAL:\n  - " + "\n  - ".join(fail))
print("[k3-dcp8pp2] verified: PP refusal gone, draft PP pinned, aux-over-PP on, "
      "block-table guard in, K3 DCP + partial-hit present")

import vllm
print(f"[k3-dcp8pp2] vllm {getattr(vllm, '__version__', '?')}")
PY

echo "=== k3-dcp8pp2: done ==="
