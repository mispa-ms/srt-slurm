#!/usr/bin/env bash
# Kimi-K3-NVFP4 + DSpark under PP, on a current vLLM nightly.
# =============================================================================
# The last cell of the matrix: the other three scripts give NVFP4 no-spec, MXFP4
# no-spec and MXFP4 DSpark. This is NVFP4 with speculation, which means four edits
# and three of them land in vllm/v1/worker/gpu/model_runner.py.
#
# ORDER IS THE WHOLE POINT. This workstream shipped the wrong order twice today and
# paid an hour each time:
#
#   [emptycache] applied: .../model_runner.py
#   [dspark-pp-main] FATAL: patch does not apply cleanly to this image
#
# because empty_cache had already shifted every hunk the strict --fuzz=0 applier was
# looking for. The PP change is a diff and matches by offset; the other two match by
# content. So the diff goes first, on the tree it was derived against, and the
# content-anchored edits follow and tolerate what it moved.
#
#   1. PP patch             --fuzz=0 diff, first
#   2. k3nvfp4-b200.sh      JET cache wiring, NVFP4 loader support, DCP dummy-batch
#                           fix -- all guarded, all idempotent
#   3. empty_cache          one line at whatever indent it sits at, last
#
# Step 2 is not the MXFP4 emptycache script even though that carries the same
# dummy-batch fix: it opens by chaining the MXFP4 HF shim, which exits 1 when its
# staged directory is absent, and an NVFP4 recipe does not set K3_STAGED_DIR.
#
# VERIFIED AS A CHAIN, not per step, against a9a17e70 -- the tree the current nightly
# image was built from -- with the result parsed at the end.
# =============================================================================
set -euo pipefail

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
command -v patch >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq patch; }

# ── 1. DSpark under pipeline parallelism ──────────────────────────────────────
echo "=== k3nvfp4-dspark-pp: spec-decode-under-PP, rebased onto upstream main ==="
if grep -q "The drafter is instantiated only on the last pipeline stage" "$VLLM_ROOT/vllm/config/speculative.py"; then
    echo "[k3nvfp4-dspark-pp] PP support already present; skipping"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 < /configs/patches/k3-dspark-pp-main.patch > /tmp/dspark-pp-dry.log 2>&1; then
        echo "[k3nvfp4-dspark-pp] FATAL: the PP patch does not apply to this image" >&2
        cat /tmp/dspark-pp-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-dspark-pp-main.patch
    echo "[k3nvfp4-dspark-pp] PP support applied"
fi

python3 - <<'PY'
import importlib.util, os, sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
u = open(os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")).read()
if "DSpark does not support pipeline parallelism" in u:
    sys.exit("[k3nvfp4-dspark-pp] FATAL: the upstream refusal is still present")
s = open(os.path.join(root, "vllm/config/speculative.py")).read()
if "pipeline_parallel_size=1" not in s:
    sys.exit("[k3nvfp4-dspark-pp] FATAL: draft ParallelConfig still inherits the target PP size")
k = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/model.py")).read()
if "unpack_aux_hidden_states(intermediate_tensors)" not in k:
    sys.exit("[k3nvfp4-dspark-pp] FATAL: the PP aux-unpack branch is missing")
if "elif self.start_layer in self.aux_hidden_state_layers:" not in k:
    sys.exit("[k3nvfp4-dspark-pp] FATAL: the aux-unpack insert landed in the wrong place")
print("[k3nvfp4-dspark-pp] verified: refusal gone, draft PP pinned, aux-unpack in place")
PY


# ── 2. JET cache, NVFP4 loader support, DCP dummy-batch fix ───────────────────
bash /configs/patches/vllm-container-deps-k3nvfp4-b200.sh

# ── 3. empty_cache before the draft loads ─────────────────────────────────────
echo "=== k3nvfp4-dspark-pp: release cached blocks before the draft model ==="

python3 - <<'PY'
import importlib.util, os, sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_runner.py",
)
src = open(target).read()
if "[emptycache]" in src:
    print("[emptycache] already applied: " + target)
    sys.exit(0)

CALL = "self.speculator.load_model(self.model)"
hits = [ln for ln in src.splitlines() if ln.strip() == CALL]
if len(hits) != 1:
    sys.exit(
        "[emptycache] FATAL: found %d lines calling the speculator loader; the load "
        "path moved and this needs re-deriving" % len(hits)
    )

anchor = hits[0]
pad = anchor[: len(anchor) - len(anchor.lstrip())]
addition = "\n".join(
    pad + line if line else ""
    for line in [
        "# [emptycache] The draft's MLA takes a second direct-DCP symmetric-memory",
        "# workspace from the driver, not the caching allocator, and the loader's",
        "# freed staging buffers leave the driver with nothing.",
        "free_before, total = torch.cuda.mem_get_info()",
        "torch.cuda.empty_cache()",
        "free_after, _ = torch.cuda.mem_get_info()",
        "logger.info(",
        '    "[emptycache] driver-visible free before draft load: "',
        '    "%.2f -> %.2f GiB of %.2f GiB",',
        "    free_before / 2**30, free_after / 2**30, total / 2**30,",
        ")",
    ]
) + "\n" + anchor

patched = src.replace(anchor, addition, 1)
compile(patched, target, "exec")
open(target, "w").write(patched)
print("[emptycache] applied at indent %d: %s" % (len(pad), target))
PY

echo "=== k3nvfp4-dspark-pp: done ==="
