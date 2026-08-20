#!/usr/bin/env bash
# Kimi-K3 GB300 disagg, Hanjie's path, on oci-aga.
# =============================================================================
# This is InferenceMAX PR #213 run on our cluster. It exists because our own
# 521-line setup script and the 57 files behind it were, for three weeks, the
# only thing standing between us and his 73.9% external prefix hit -- and none
# of it was the vLLM patch. That was checked rather than assumed: his
# hjjq/k3-patches@0120f353 and our two patches, applied separately to the same
# nightly 3d204dfda, produce byte-identical vllm trees. So the code is not the
# variable and this script does not try to be clever about it.
#
# WHAT IT DOES, in full:
#   1. resolve the staged K3 checkpoint (oci-aga has no /scratch/models alias)
#   2. run Hanjie's applier, unmodified
#
# WHAT IT DELIBERATELY DOES NOT DO, all of which our script did:
#   - install or configure Mooncake. The image already carries 0.3.12.post1 for
#     CUDA 13, and srtctl renders store_config and starts ONE master for the job
#     when the recipe has a mooncake_kv_store block. Ours started one per node,
#     which is four independent stores in 1P1D, which is why an 8-way lookup
#     quorum could never be satisfied and the external hit read 0.0% for weeks.
#   - reinstall flashinfer. Hanjie takes the image's build.
#   - apt-get anything.
#   - apply anything of ours on top of his patch.
#
# Every one of those is a deviation we would have to price separately if the
# number does not land, so none of them are here on the first attempt.
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
            echo "[k3] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3] FATAL: no staged checkpoint on this cluster. Set K3_STAGED_DIR." >&2
    echo "[k3] Refusing to continue -- the fallback is a 1.45 TB download inside" \
         "a 4-hour job, which fails later and less legibly than this does." >&2
    exit 1
fi

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/apply-vllm-kimi-k3-dcp-aug13.sh

# Say what the run is actually made of, so the log answers "was this the strict
# port?" without anyone having to reconstruct it from the config.
echo "=== kimi-k3-hanjie-strict: image mooncake, srtctl master, Hanjie patch only ==="
python3 - <<'PY'
import importlib.util, os
spec = importlib.util.find_spec("mooncake")
print("[strict] mooncake:", os.path.dirname(spec.origin) if spec else "ABSENT")
try:
    import mooncake.store  # noqa: F401
    print("[strict] mooncake.store importable")
except Exception as exc:
    print("[strict] mooncake.store import FAILED:", exc)
for var in ("UCX_TLS", "UCX_NET_DEVICES", "MOONCAKE_MASTER", "MOONCAKE_CONFIG_PATH"):
    print(f"[strict] {var}={os.environ.get(var, '<unset>')}")
PY

# --- nsys, appended for the profiling arms -------------------------------
#
# The base script above is the strict Hanjie port and nothing else; this adds
# Nsight Systems, which vllm/vllm-openai does not ship. Without it the wrapped
# worker launch exits 127 about a minute in, which reads as a worker crash
# rather than a missing binary.
#
# Failing here rather than there is the point: if nsys is still off PATH after
# the install, stop at setup with that sentence in the log.
bash /configs/patches/install-nsys-cli.sh

if ! command -v nsys >/dev/null 2>&1; then
  echo "[strict-nsys] FATAL: nsys is not on PATH after install-nsys-cli.sh" >&2
  exit 1
fi
echo "[strict-nsys] $(nsys --version | head -1)"
