#!/usr/bin/env bash
# Kimi-K3 HF cache shim plus a FlashInfer upgrade to 0.6.16rc5.
# =============================================================================
# WHY: vLLM main refuses the TRT-LLM MXFP4 MoE path for this model --
#
#   ValueError: Mxfp4 MoE backend 'FLASHINFER_TRTLLM_MXFP4_MXFP8' does not support the
#   deployment configuration since kernel does not support MoEActivation.SITU activation.
#
# Kimi-K3 declares text_config.hidden_act = "situ" (activation_situ_beta 4.0), so
# moe-backend: auto falls back to DEEPGEMM_MXFP4 and the run is 17-20% slower than the day-0
# vllm/vllm-openai:kimi-k3 image, which selects FLASHINFER_TRTLLM_MXFP4_MXFP8 /
# TrtLlmMxfp4Experts. Measured at c16 without speculation, n=3 per container, no overlap
# between the sets, exact permutation p=0.050:
#     day-0 kimi-k3    3,535 tok/s/GPU
#     nightly 07-31    2,931   (-17.1%)
#     nightly 08-03    2,822   (-20.2%)
#
# The refusal is not a hardware limit. FlashInfer v0.6.16 (2026-07-31) carries
#   cherry-pick: #4180 ([feat] Add SITU trtllmgen MOE)  -- flashinfer-ai/flashinfer#4252
# and the nightly images still ship 0.6.15.post1, which predates it. So the regression may be
# the bundled FlashInfer rather than vLLM itself.
#
# Both containers produce byte-identical output on a six-prompt greedy probe, so this is a
# performance question, not a correctness one.
#
# WHY FROM GITHUB RELEASES: flashinfer-python is on PyPI at 0.6.16, but flashinfer-cubin stops
# at 0.6.13 there and flashinfer-jit-cache is absent; neither is on pypi.nvidia.com,
# wheels.vllm.ai or download.pytorch.org. The release page carries all three as assets. The
# trtllm-gen MoE kernels live in the cubin package, so upgrading flashinfer-python alone would
# not bring the SITU kernel.
#
# rc5 rather than 0.6.16 / 0.6.16.post1 on purpose: it is what the K3 DISAGG stack on this
# branch already runs, and it is the first release carrying the MegaMoE kernels. Keeping the
# two tracks on one FlashInfer means an AGG number and a DISAGG number stay comparable.
#
# ~2.5 GB of wheels. Comparable to the 421 MB nsys package the sibling script already pulls.
# =============================================================================
set -euo pipefail

FI_VERSION="0.6.16rc5"
FI_BASE="https://github.com/flashinfer-ai/flashinfer/releases/download/v${FI_VERSION}"

echo "=== k3-flashinfer: upgrading FlashInfer to ${FI_VERSION} ==="
python3 -c "import flashinfer, sys; print(f'[k3-fi] before: {flashinfer.__version__}')" 2>/dev/null \
    || echo "[k3-fi] before: flashinfer not importable"

# --no-deps so the pinned torch/vllm stack in the image is left alone.
pip install --no-deps --force-reinstall \
    "${FI_BASE}/flashinfer_python-${FI_VERSION}-py3-none-any.whl" \
    "${FI_BASE}/flashinfer_cubin-${FI_VERSION}-py3-none-any.whl" \
    "${FI_BASE}/flashinfer_jit_cache-${FI_VERSION}+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"

# Fail here rather than 40 minutes later inside the engine.
python3 - <<'FI_EOF'
import sys
import flashinfer
print(f"[k3-fi] after: {flashinfer.__version__}")
if not flashinfer.__version__.startswith("0.6.16"):
    print(f"[k3-fi] FATAL: expected 0.6.16.x, got {flashinfer.__version__}", file=sys.stderr)
    sys.exit(1)
FI_EOF
echo "=== k3-flashinfer: done ==="

set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps.sh ]]; then
    bash /configs/patches/vllm-container-deps.sh
fi

REPO_ID="moonshotai/Kimi-K3"
STAGED_DIR="${K3_STAGED_DIR:-/lustre/share/coreai_comparch_aarwlt/hf_repos/moonshotai/Kimi-K3}"
# HEAD of moonshotai/Kimi-K3 as of 2026-07-27; only used if the Hub is unreachable.
FALLBACK_SHA="9f62e4e9fffbd0a83ddd60e1c209d828994b3569"

echo "=== k3-hfshim: wiring HF cache to the pre-staged K3 checkpoint ==="

if [[ -z "${HF_HOME:-}" ]]; then
    echo "[k3-hfshim] FATAL: HF_HOME is not set; cannot place the cache entry." >&2
    exit 1
fi

if [[ ! -d "$STAGED_DIR" ]]; then
    echo "[k3-hfshim] FATAL: staged checkpoint not found at $STAGED_DIR" >&2
    echo "[k3-hfshim] Refusing to continue — HF would fall back to a ~1.4 TB download." >&2
    echo "[k3-hfshim] Check the per-cluster staging status, or override K3_STAGED_DIR." >&2
    exit 1
fi

CACHE_ENTRY="$HF_HOME/hub/models--moonshotai--Kimi-K3"

if [[ -d "$STAGED_DIR/snapshots" ]]; then
    # The staged copy is already a full HF cache entry — point at it wholesale.
    echo "[k3-hfshim] staged copy is HF-cache-shaped; linking $CACHE_ENTRY -> $STAGED_DIR"
    mkdir -p "$HF_HOME/hub"
    ln -sfn "$STAGED_DIR" "$CACHE_ENTRY"
else
    # Plain repo snapshot (config.json + weights at the top level).
    if [[ ! -f "$STAGED_DIR/config.json" ]]; then
        echo "[k3-hfshim] FATAL: $STAGED_DIR has neither snapshots/ nor config.json." >&2
        echo "[k3-hfshim] The staging copy looks incomplete — is the download still running?" >&2
        exit 1
    fi

    SHA="$(git ls-remote "https://huggingface.co/$REPO_ID" HEAD 2>/dev/null | awk '{print $1}' | head -1 || true)"
    if [[ -z "$SHA" ]]; then
        SHA="$FALLBACK_SHA"
        echo "[k3-hfshim] Hub unreachable; using pinned sha $SHA"
    else
        echo "[k3-hfshim] resolved $REPO_ID main -> $SHA"
    fi

    mkdir -p "$CACHE_ENTRY/refs" "$CACHE_ENTRY/snapshots"
    ln -sfn "$STAGED_DIR" "$CACHE_ENTRY/snapshots/$SHA"
    # Atomic so a concurrent reader never sees a half-written ref.
    printf '%s' "$SHA" > "$CACHE_ENTRY/refs/.main.$$"
    mv -f "$CACHE_ENTRY/refs/.main.$$" "$CACHE_ENTRY/refs/main"
    echo "[k3-hfshim] cache entry ready: $CACHE_ENTRY (snapshots/$SHA -> $STAGED_DIR)"
fi

# Prove the wiring works with no network at all. If this fails the job would have
# started a 1.4 TB download, so fail here instead where the message is obvious.
python3 - <<'PY'
import sys
from huggingface_hub import snapshot_download

try:
    path = snapshot_download("moonshotai/Kimi-K3", local_files_only=True)
except Exception as exc:
    print(f"[k3-hfshim] FATAL: offline resolution failed: {exc}", file=sys.stderr)
    sys.exit(1)
print(f"[k3-hfshim] offline resolution OK -> {path}")
PY

echo "=== k3-hfshim: done ==="
