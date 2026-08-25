#!/usr/bin/env bash
# Kimi-K3-NVFP4: point the HF cache at the JET-managed checkpoint.
# =============================================================================
# Sibling of vllm-container-deps-k3-hfshim.sh, which does the same job for the
# BF16 moonshotai/Kimi-K3. Two differences, both consequences of the checkpoint
# coming from JET rather than from a copy we staged ourselves:
#
#   1. The source is JET's artifact store, not our project's scratch. JET already
#      syncs nvidia/Kimi-K3-NVFP4 to lyris, oci-aga, prenyx and nsc-svg; there is
#      no reason to hold a second 1.6 TB copy under coreai_comparch_inferencex.
#      The tree is owned by svc-jet:jet-admins and is world-readable.
#
#   2. The JET path root differs per cluster (/lustre/share on lyris/prenyx,
#      /scratch/fsw/portfolios/coreai/projects on oci-aga/nsc-svg), so the recipe
#      bind-mounts the artifact family at a fixed container path and STAGED_DIR
#      defaults to that. One shim, every cluster, no probing.
#
# WHY A SHIM AT ALL. A JET path is a plain repo snapshot -- config.json and the 96
#   shards at the top level. It is not HF-cache-shaped, so HF_HOME cannot simply be
#   pointed at it. AIB requires model.path to be "hf:<org>/<model>" for portability,
#   so the model resolves through the HF cache, and this materialises the cache
#   layout with symlinks:
#       models--nvidia--Kimi-K3-NVFP4/refs/main       -> <commit sha>
#       models--nvidia--Kimi-K3-NVFP4/snapshots/<sha> -> the JET directory
#   Only $HF_HOME is written. The JET tree is read-only to us and stays untouched.
#
# REVISION. Pinned, and the Hub is deliberately not consulted. Asking it at run time
#   is what broke every K3 arm on 2026-08-20: moonshotai pushed a commit adding four
#   files, and the staged directory was then presented under a sha whose file list it
#   could not satisfy -- IncompleteSnapshotError, eight arms, an hour each. The pin
#   below is JET artifact model/nvidia_kimi-k3-nvfp4/hf:hf-f8c5234_orig, whose file
#   list and per-file sizes were checked against the Hub on 2026-08-25: 116 files,
#   96 shards, 1,610,039,528,867 B, exact match. Re-stage and move the pin together.
#
# The shim is idempotent and safe to run concurrently: every write is mkdir -p /
#   ln -sfn / atomic rename. It refuses to run if the checkpoint is absent, because
#   the alternative is silently burning job walltime on a 1.6 TB download.
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps.sh ]]; then
    bash /configs/patches/vllm-container-deps.sh
fi

REPO_ID="nvidia/Kimi-K3-NVFP4"
# Container path. The recipe bind-mounts the JET artifact family here; see the
# extra_mount block in the sweep config for the per-cluster host path.
STAGED_DIR="${K3NVFP4_STAGED_DIR:-/jet-k3nvfp4/hf/hf-f8c5234_orig}"
FALLBACK_SHA="f8c5234a0a880bcc6cbf779a315e7ee2f405b812"

echo "=== k3nvfp4-hfshim: wiring HF cache to the JET checkpoint ==="

# HF_HOME is set in the recipe's aggregated_environment, but it does not always reach
# this script: three GB300 runs (oci-aga 566436, lyris 2788759, and 64537688) died here
# with it empty while the patched config plainly carried it. Rather than block on that,
# fall back to the directory an env-less huggingface_hub would use anyway -- the point
# is that the worker and this script agree on where the cache is, not that the recipe
# was the one to say so. Loud, because a run that silently used the wrong cache would
# be worse than one that stopped.
if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME="${HOME:-/root}/.cache/huggingface"
    echo "[k3nvfp4-hfshim] WARNING: HF_HOME was not in this script's environment." >&2
    echo "[k3nvfp4-hfshim] WARNING: falling back to $HF_HOME, which is where a worker" >&2
    echo "[k3nvfp4-hfshim] WARNING: without HF_HOME will look. The recipe sets HF_HOME;" >&2
    echo "[k3nvfp4-hfshim] WARNING: that it is missing here is a separate bug worth chasing." >&2
fi
mkdir -p "$HF_HOME"

if [[ ! -d "$STAGED_DIR" ]]; then
    echo "[k3nvfp4-hfshim] FATAL: checkpoint not found at $STAGED_DIR" >&2
    echo "[k3nvfp4-hfshim] Refusing to continue — HF would fall back to a ~1.6 TB download." >&2
    echo "[k3nvfp4-hfshim] Check the extra_mount in the recipe, or override K3NVFP4_STAGED_DIR." >&2
    echo "[k3nvfp4-hfshim] Resolve the live path with: jet_model.py path $REPO_ID --cluster <c>" >&2
    exit 1
fi

# A mount that exists but is empty is the failure this catches: the bind succeeded
# against a path that does not hold the artifact.
if [[ ! -f "$STAGED_DIR/config.json" ]]; then
    echo "[k3nvfp4-hfshim] FATAL: $STAGED_DIR has no config.json." >&2
    echo "[k3nvfp4-hfshim] The mount points somewhere that is not the snapshot root." >&2
    ls -la "$STAGED_DIR" 2>&1 | head -10 >&2 || true
    exit 1
fi

SHARDS=$(ls "$STAGED_DIR"/*.safetensors 2>/dev/null | wc -l)
if [[ "$SHARDS" -ne 96 ]]; then
    echo "[k3nvfp4-hfshim] FATAL: expected 96 safetensors shards, found $SHARDS." >&2
    echo "[k3nvfp4-hfshim] A partial checkpoint fails later and less legibly than here." >&2
    exit 1
fi

SHA="${K3NVFP4_HFSHIM_SHA:-$FALLBACK_SHA}"
CACHE_ENTRY="$HF_HOME/hub/models--nvidia--Kimi-K3-NVFP4"
echo "[k3nvfp4-hfshim] $SHARDS shards at $STAGED_DIR, labelling as $SHA"

mkdir -p "$CACHE_ENTRY/refs" "$CACHE_ENTRY/snapshots"
ln -sfn "$STAGED_DIR" "$CACHE_ENTRY/snapshots/$SHA"
# Atomic so a concurrent reader never sees a half-written ref.
printf '%s' "$SHA" > "$CACHE_ENTRY/refs/.main.$$"
mv -f "$CACHE_ENTRY/refs/.main.$$" "$CACHE_ENTRY/refs/main"
echo "[k3nvfp4-hfshim] cache entry ready: $CACHE_ENTRY (snapshots/$SHA -> $STAGED_DIR)"

# Prove the wiring works with no network at all. If this fails the job would have
# started a 1.6 TB download, so fail here instead where the message is obvious.
python3 - <<'PY'
import sys
from huggingface_hub import snapshot_download

try:
    path = snapshot_download("nvidia/Kimi-K3-NVFP4", local_files_only=True)
except Exception as exc:
    print(f"[k3nvfp4-hfshim] FATAL: offline resolution failed: {exc}", file=sys.stderr)
    sys.exit(1)
print(f"[k3nvfp4-hfshim] offline resolution OK -> {path}")
PY

echo "=== k3nvfp4-hfshim: done ==="
