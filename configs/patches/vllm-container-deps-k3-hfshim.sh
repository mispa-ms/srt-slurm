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
# REVISION: the staged copy is the source of truth and is labelled with a fixed sha
#   (FALLBACK_SHA, overridable via K3_HFSHIM_SHA). The Hub is deliberately not asked at
#   run time. Asking it was the old behaviour and it broke every K3 job within minutes
#   of upstream publishing a commit, because the staged directory then carried a label
#   whose file list it could not satisfy. Re-stage and move the pin together.
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps.sh ]]; then
    bash /configs/patches/vllm-container-deps.sh
fi

REPO_ID="moonshotai/Kimi-K3"
STAGED_DIR="${K3_STAGED_DIR:-/lustre/share/coreai_comparch_aarwlt/hf_repos/moonshotai/Kimi-K3}"
# The revision the shared copy was staged from, and the one every K3 number on this
# workstream was measured under. This is the pin, not a fallback -- see the note at the
# labelling step for why the Hub is no longer asked.
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

    # Label the staged copy with the revision it actually IS, not with whatever the
    # Hub happens to be serving now.
    #
    # This used to read HEAD from the Hub every job. That works only while upstream is
    # frozen, and it broke the moment it was not: moonshotai pushed a590ce09 on
    # 2026-08-20T04:57Z adding four .eval_results/*.yaml, and from the next job onward
    # every K3 arm died in the offline check below --
    #
    #   IncompleteSnapshotError: the cached snapshot for 'moonshotai/Kimi-K3'
    #   (revision 'main', commit a590ce09...) is incomplete: 12 file(s) are missing
    #
    # -- because the staged directory was being presented under a sha whose file list
    # it does not satisfy. Nothing was wrong with the weights; the label was a lie.
    # Eight arms lost an hour each to it.
    #
    # So the pin is now the default and the Hub is not consulted. Set K3_HFSHIM_SHA
    # after re-staging to move it; the offline check below fails loudly if the pin and
    # the staged contents ever disagree, which is the failure we want.
    SHA="${K3_HFSHIM_SHA:-$FALLBACK_SHA}"
    echo "[k3-hfshim] using pinned sha $SHA (staged copy is the source of truth, not the Hub)"

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
