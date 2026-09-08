#!/usr/bin/env bash
# Apply upstream vllm#55472 verbatim: preserve the target parallel config for the
# DSpark draft (PP=1, TP=draft, everything else -- DCP included -- inherited).
# =============================================================================
# This is the upstream fix for the second 09/08 wall, applied as-is so a review
# on #55472 can say "this diff, on 9ea8f3ff, on B200 TP8xPP2xDCP8: PP1 passes".
# Our own stopgap for the same wall, vllm-container-deps-k3-dspark-draft-dcp.sh,
# rewrites the SAME `parallel_config=` block from the other direction; the two
# conflict (verified: dcp-then-55472 fails hunk #1), so a chain carries one or
# the other, never both.
#
# THE WALL (recap). PP1 alone, 6 s after the draft loads, in profiling KV-cache
# init: mla_attention.py:2357 `assert isinstance(self.dcp_manager, MLADCPManager)`.
# The metadata builder reads DCP from the process group (8); the layer's impl
# reads it from the config (mla_attention.py:3213); #50514 gave the draft
# create_draft_parallel_config(), which never sets decode_context_parallel_size,
# so the draft impl sees 1 and builds no manager. 6f7df92a inherited the target's
# config wholesale and was fine. #55472 (starkwj, 2026-09-05) restores that.
#
# Order matters: vllm-container-deps-k3-dspark-draft-loader.sh must run first.
# #55472 applies after it at a one-line offset (verified with --fuzz=0).
# =============================================================================
set -euo pipefail

echo "=== dspark-pr55472: upstream #55472 verbatim ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
DU="$VLLM_ROOT/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
[ -f "$DU" ] || { echo "[pr55472] FATAL: no dspark/utils.py in this image" >&2; exit 1; }

python3 - "$DU" <<'PY'
import sys
src = open(sys.argv[1]).read()
flat = "".join(src.split())
if "parallel_config=replace(vllm_config.parallel_config,pipeline_parallel_size=1" in flat:
    print("[pr55472] already present in this image"); sys.exit(10)
if "decode_context_parallel_size=(vllm_config.parallel_config.decode_context_parallel_size)" in flat:
    sys.exit("[pr55472] FATAL: the draft-dcp stopgap is already applied here -- the two rewrite the same block; drop one from the chain")
PY
rc=$?
if [ "$rc" -eq 10 ]; then
    :
elif [ "$rc" -ne 0 ]; then
    exit "$rc"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-pr55472.patch > /tmp/pr55472-dry.log 2>&1; then
        echo "[pr55472] FATAL: #55472 does not apply to this image" >&2
        echo "[pr55472] cut against 9ea8f3ff after the draft-loader guard; check both" >&2
        cat /tmp/pr55472-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-pr55472.patch
    echo "[pr55472] applied under $VLLM_ROOT"
fi

# Verify the shape the review will claim, and that the loader guard survived.
python3 - "$DU" <<'PY'
import ast, sys
src = open(sys.argv[1]).read()
flat = "".join(src.split())
if "parallel_config=replace(vllm_config.parallel_config,pipeline_parallel_size=1" not in flat:
    sys.exit("[pr55472] FATAL: the target-config-with-PP=1 override is not present after applying")
if "get_pp_group().world_size>1" not in flat:
    sys.exit("[pr55472] FATAL: the draft-loader guard is gone -- chain order is wrong")
bound = set()
for n in ast.walk(ast.parse(src)):
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            bound.add(a.asname or a.name.split(".")[0])
for name in ("replace", "get_pp_group"):
    if name not in bound:
        sys.exit(f"[pr55472] FATAL: `{name}` is used but not imported")
print("[pr55472] verified: target parallel config preserved with PP=1; loader guard intact; names bound")
PY

echo "=== dspark-pr55472: done ==="
