#!/usr/bin/env bash
# Kimi-K3: point the HF cache at the pre-staged shared checkpoint.
# =============================================================================
# WHY: the K3 checkpoint is ~1.4 TB. A shared copy is already staged per cluster
#   (bia: /lustre/share/coreai_comparch_aarwlt/hf_repos/moonshotai/Kimi-K3,
#   permission: all), but AIB requires model.path to be "hf:<org>/<model>" for
#   cross-cluster portability, so srt-slurm resolves the model through the HF
#   cache instead of bind-mounting a directory. Without this shim HF would not
#   find the staged copy and would try to pull 1.4 TB inside the job.
#
# FIX: materialize the standard HF cache layout under $HF_HOME/hub so that
#   "moonshotai/Kimi-K3" resolves to the staged directory:
#       models--moonshotai--Kimi-K3/refs/main       -> <commit sha>
#       models--moonshotai--Kimi-K3/snapshots/<sha> -> symlink to the staged dir
#   The snapshot is named with the REAL commit sha (not a synthetic revision) so
#   that online resolution matches too — with a synthetic name the Hub would
#   report a different sha for main and HF would download anyway.
#
# The shim is idempotent and safe to run concurrently: every write is mkdir -p /
#   ln -sfn / atomic rename. It refuses to run if the staged copy is absent or
#   incomplete, because the alternative is silently burning job walltime on a
#   1.4 TB download.
#
# CAVEAT: the sha is read from the Hub at run time and the staged directory is
#   then labelled with it. If upstream publishes a new commit after the copy was
#   staged, this presents stale weights under the new sha rather than
#   re-downloading — deliberate (a 1.4 TB pull inside a benchmark job is worse),
#   but it means the staged copy is the source of truth, not the Hub. Re-stage
#   when the upstream repo changes.
# =============================================================================
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
