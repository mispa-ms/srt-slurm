#!/usr/bin/env bash
# Kimi-K3 GB300 wide-EP on the aug17 nightly, plus Nsight Systems.
# =============================================================================
# Same stack as kimi-k3-hjstrict-fix50359.sh -- staged checkpoint, HF shim,
# Wei's agentx-v2 runtime patch, vllm#50359 -- moved from the aug13 nightly to
# the aug17 one.
#
# WHY THE MOVE. Docker Hub deleted the aug13 tag. `nightly-3d204dfda...` now
# 404s, and an arm on a missing image dies at
#     RuntimeError: NATS failed to start
# because the infra container is launched from the job image and srun exits 1
# before any of our code runs. Three arms went that way (pipeline 63567651) on
# an image that 94 previous arms had used. **Nightly tags are garbage-collected;
# check the tag exists before submitting, and treat "NATS failed to start" as
# "the image is gone" until proven otherwise.**
#
# WHAT CHANGES WITH THE NIGHTLY, and what does not. The K3 runtime patch is
# Hanjie's own aug17 rebuild of the same wzhao/kimi-k3-agentx-v2@face29e65
# content, so the vLLM tree is the same code on a newer base. vllm#50359 is
# unchanged: it touches only mooncake/store/coordinator.py, which is byte
# identical (17,102 bytes) across the two nightlies.
#
# NOT CARRIED: the mooncake load-failure scan. It is logging-only and it was
# there to diagnose the -704 failures, which are closed.
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
bash /configs/patches/apply-vllm-pr50359.sh

# Say what the run is made of. The EP path is new to this track, so print the
# two things that decide whether an EP arm can start at all before it spends
# twenty minutes loading: whether DeepGEMM's mega_moe symbols import on aarch64,
# and whether torch symmetric memory is there for the cross-node rendezvous.
echo "=== kimi-k3-wideep-aug17: agentx-v2 on nightly-311b3513a, +pr50359 ==="
python3 - <<'PY'
import os

try:
    import vllm.third_party.deep_gemm as dg
    have = all(hasattr(dg, n) for n in ("fp8_fp4_mega_moe", "get_symm_buffer_for_mega_moe"))
    print(f"[k3-wideep] deep_gemm imported; mega_moe symbols present: {have}")
except Exception as exc:
    print(f"[k3-wideep] deep_gemm import FAILED: {exc}")

try:
    import torch.distributed._symmetric_memory as symm_mem  # noqa: F401
    print("[k3-wideep] torch symmetric memory importable")
except Exception as exc:
    print(f"[k3-wideep] torch symmetric memory import FAILED: {exc}")

for var in ("UCX_TLS", "UCX_NET_DEVICES", "NCCL_MNNVL_ENABLE", "MOONCAKE_MASTER"):
    print(f"[k3-wideep] {var}={os.environ.get(var, '<unset>')}")
PY

# --- nsys, for the profiling arms ----------------------------------------
#
# vllm/vllm-openai ships no nsys. Without it the wrapped worker launch exits
# 127 about a minute in, which reads as a worker crash rather than a missing
# binary -- so fail here, with that sentence, instead of there.
bash /configs/patches/install-nsys-cli.sh

if ! command -v nsys >/dev/null 2>&1; then
  echo "[k3-wideep-nsys] FATAL: nsys is not on PATH after install-nsys-cli.sh" >&2
  exit 1
fi
echo "[k3-wideep-nsys] $(nsys --version | head -1)"
