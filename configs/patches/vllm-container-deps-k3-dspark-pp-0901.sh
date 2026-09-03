#!/usr/bin/env bash
# spec-decode-under-PP for the 2026-09-01 nightly (7c5dc571).
# =============================================================================
# Same carry as vllm-container-deps-k3-dspark-pp-828.sh, minus one hunk. This is
# NOT a re-derivation: vllm#50514, which would land this upstream, is still open,
# so the 596 lines stay ours and we re-target them rather than rewrite them.
#
# The dropped hunk rewrote the `not_finishing` term in
# vllm/v1/worker/gpu/pp_utils.py. It existed because the scheduler advances
# num_computed_tokens by the full scheduled width up front and only rolls the
# rejected part back in update_from_output, which under PP runs after the next
# batch is already scheduled -- so comparing the inflated count against
# max_seq_len marked a request as finishing up to num_draft tokens early.
#
# Upstream 6bafc049aa "[Bugfix][PP] Never drop a decoding request from the
# sampled-token broadcast" fixed the same bug more robustly: it removed the
# max_seq_len test from that function outright, so there is nothing left to
# correct. The other five hunks still apply, with offsets.
# =============================================================================
set -euo pipefail

echo "=== dspark-pp-0901: spec-decode-under-PP for 7c5dc571 ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')

if grep -q "The drafter is instantiated only on the last pipeline stage" \
     "$VLLM_ROOT/vllm/config/speculative.py"; then
    echo "[dspark-pp-0901] already present in this image"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-dspark-pp-0901.patch > /tmp/dspp0901-dry.log 2>&1; then
        echo "[dspark-pp-0901] FATAL: does not apply to this image" >&2
        echo "[dspark-pp-0901] cut against 7c5dc571; on 6f7df92a use the -828 patch" >&2
        cat /tmp/dspp0901-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-dspark-pp-0901.patch
    echo "[dspark-pp-0901] applied under $VLLM_ROOT"
fi

# Verify the two things the speculative arms actually depend on: that the drafter
# is built on the last stage, and that the aux-hidden-state path is PP-aware. A
# patch that applied but left either out would degrade acceptance silently, and
# under synthetic rejection the arm would still report the AL we fed it.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
spec = open(os.path.join(root, "vllm/config/speculative.py")).read()
iface = open(os.path.join(root, "vllm/model_executor/models/interfaces.py")).read()

if "The drafter is instantiated only on the last pipeline stage" not in spec:
    sys.exit("[dspark-pp-0901] FATAL: the last-stage drafter comment is missing")
if "supports_aux_hidden_states_over_pp" not in iface:
    sys.exit("[dspark-pp-0901] FATAL: supports_aux_hidden_states_over_pp is missing")
print("[dspark-pp-0901] verified: last-stage drafter and PP aux-hidden-states present")
PY

echo "=== dspark-pp-0901: done ==="
