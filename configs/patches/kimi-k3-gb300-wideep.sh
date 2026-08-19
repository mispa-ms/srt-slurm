#!/usr/bin/env bash
# Kimi-K3 GB300 AGG wide-EP ladder, on oci-aga.
# =============================================================================
# This is the setup for the EP-width study: TP4 held constant, DP (and so EP)
# swept 8 -> 16 -> 32, per-engine batch held at 64 by the prefix-replay client's
# X-data-parallel-rank round-robin. The question it serves is Hanjie's -- why
# DEP16 is bad when DLSim projects it should win -- and it is deliberately an
# AGG study, because srtctl's own prefix-replay runner refuses a disaggregated
# backend (src/srtctl/benchmarks/prefix_replay.py: "prefix-replay requires an
# aggregated backend"). Decode cannot be isolated across a P/D split.
#
# WHAT IT DOES, in full:
#   1. resolve the staged K3 checkpoint (oci-aga has no /scratch/models alias)
#   2. wire the HF cache at it, so "hf:moonshotai/Kimi-K3" resolves offline
#   3. run Hanjie's aug17 applier, unmodified
#
# WHAT IT DELIBERATELY DOES NOT DO:
#   - install or configure Mooncake. There is no offload tier in this study and
#     no KV connector at all; AGG has neither NIXL nor a store. Every one of the
#     three faults that cost the disagg track three days lives in that plumbing.
#   - reinstall flashinfer. The image's build is the measurement.
#   - apply anything of ours on top of Hanjie's patch. His aug17 applier already
#     covers "the Kimi-K3 DCP, DEP, hybrid-cache, speculative-decoding, and
#     Mooncake paths used by this study" -- DEP included, in his own words.
#
# The applier refuses any image whose _version.py is not g311b3513a, so the
# container tag is load-bearing:
#   vllm/vllm-openai:nightly-311b3513af33bc29b4acb2fde2e9313e5e9966a0
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
            echo "[k3-wideep] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3-wideep] FATAL: no staged checkpoint on this cluster. Set K3_STAGED_DIR." >&2
    echo "[k3-wideep] Refusing to continue -- the fallback is a 1.45 TB download inside" \
         "a 4-hour job, which fails later and less legibly than this does." >&2
    exit 1
fi

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/apply-vllm-kimi-k3-aug17.sh

# Say what the run is made of, so the log answers the three questions that
# decide whether an arm is readable before it has produced a number:
#   - is the DeepGEMM extension present at all on aarch64? The vendored tree
#     carries only x86_64 _C .so files in the source checkout, so mega_moe on
#     GB300 is unproven for us and this is where it shows.
#   - does torch symmetric memory exist? mega_moe rendezvouses a symm buffer
#     across the whole EP group, which at EP16/EP32 spans 4 and 8 nodes.
#   - what did the EP knobs actually resolve to?
echo "=== kimi-k3-gb300-wideep: image flashinfer, Hanjie aug17 patch, no store ==="
python3 - <<'PY'
import os

try:
    import vllm.third_party.deep_gemm as dg
    have_mega = all(hasattr(dg, n) for n in ("fp8_fp4_mega_moe", "get_symm_buffer_for_mega_moe"))
    print(f"[k3-wideep] deep_gemm imported; mega_moe symbols present: {have_mega}")
except Exception as exc:
    print(f"[k3-wideep] deep_gemm import FAILED: {exc}")

try:
    import torch.distributed._symmetric_memory as symm_mem  # noqa: F401
    print("[k3-wideep] torch symmetric memory importable")
except Exception as exc:
    print(f"[k3-wideep] torch symmetric memory import FAILED: {exc}")

for var in (
    "UCX_TLS", "UCX_NET_DEVICES", "NCCL_MNNVL_ENABLE", "NCCL_NVLS_ENABLE",
    "NCCL_IB_HCA", "PYTORCH_ALLOC_CONF", "DG_JIT_CACHE_DIR",
):
    print(f"[k3-wideep] {var}={os.environ.get(var, '<unset>')}")
PY
