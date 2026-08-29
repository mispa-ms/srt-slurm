#!/usr/bin/env bash
# DSpark under PP3: the dspark-pp chain, plus an opt-in lift of its own pp>2 refusal.
# =============================================================================
# WHAT THE REFUSAL IS. k3-dspark-pp.patch enables DSpark under pipeline parallelism
# and then caps it at pp=2 itself. The reason is in its own comment: the aux-tap
# accounting is size-agnostic and unit-tested to pp=8, but a *middle* stage -- one
# that both adopts upstream taps and contributes its own -- has never run on
# hardware, and this feature fails as degraded acceptance rather than a crash.
#
# WHY LIFT IT NOW. Everything except the hardware check is already in hand:
#
#   - tests/v1/worker/test_eagle3_aux_hidden_states_pp.py parametrises pp=1,2,3,4,6,8
#   - the accounting was re-derived for the exact shape K3 runs -- 93 layers,
#     VLLM_PP_LAYER_PARTITION=32,32,29, DSpark taps [2,23,47,71,89]. rank 1 is the
#     middle stage: 2 taps in, its own 47 added, 3 forwarded, and the drafter ends up
#     with all five in order. rank 0 sends 2 and rank 2 expects 3, so send and recv
#     agree at both boundaries.
#
# So the missing piece is exactly the acceptance measurement the refusal asks for, and
# that measurement needs a server that starts.
#
# WHY AN ENV VAR AND NOT A DELETION. Under rejection_sample_method=synthetic the
# acceptance length is an *input*: a broken middle stage would still report the AL the
# recipe fed it. A silent bypass would therefore manufacture exactly the number the
# refusal exists to prevent. VLLM_DSPARK_PP_ALLOW_UNVALIDATED=1 has to be set on the
# arm, it is logged at WARNING on every start, and the refusal text is left in place
# for anyone who has not set it. Runs that use it must be standard-rejection until the
# comparison is done.
#
# HOW TO RETIRE THIS SCRIPT. Run ns=7 at c1 on pp=2 and pp=3 with
# rejection_sample_method=standard and compare measured acceptance. If they agree, the
# middle stage is proven and the cap in k3-dspark-pp*.patch can be raised for good --
# at which point this file and its patch are deleted rather than kept as a flag.
#
# The hunk anchors only on lines k3-dspark-pp.patch inserts, and that block is
# byte-identical in the 728d3ad and 08-28 variants, so this applies on either chain.
# =============================================================================
set -euo pipefail

echo "=== dspark-pp3: opt-in lift of the pp>2 refusal ==="

bash /configs/patches/vllm-container-deps-k3-b200-dspark-pp.sh

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')

if grep -q "VLLM_DSPARK_PP_ALLOW_UNVALIDATED" \
     "$VLLM_ROOT/vllm/v1/worker/gpu/model_runner.py"; then
    echo "[dspark-pp3] already present in this image"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-dspark-pp3-guard-only.patch > /tmp/dspp3-dry.log 2>&1; then
        echo "[dspark-pp3] FATAL: does not apply to this image" >&2
        cat /tmp/dspp3-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-dspark-pp3-guard-only.patch
    echo "[dspark-pp3] applied under $VLLM_ROOT"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
mr = open(os.path.join(root, "vllm/v1/worker/gpu/model_runner.py")).read()

# The gate must exist, and the refusal must survive for anyone who has not set it --
# a patch that removed the raise outright would also break dspark-pp's own verifier.
if "VLLM_DSPARK_PP_ALLOW_UNVALIDATED" not in mr:
    sys.exit("[dspark-pp3] FATAL: the env gate is not present")
if "pipeline_parallel_size=2" not in mr:
    sys.exit("[dspark-pp3] FATAL: the pp>2 refusal text was removed, not gated")
if mr.count("raise NotImplementedError") != 1:
    sys.exit("[dspark-pp3] FATAL: expected exactly one pp>2 raise after patching")

import vllm.v1.worker.gpu.model_runner  # noqa: F401

if os.environ.get("VLLM_DSPARK_PP_ALLOW_UNVALIDATED") == "1":
    print("[dspark-pp3] gate is SET: pp>2 will be allowed and warned about")
else:
    print("[dspark-pp3] gate is not set: pp>2 still refused (patch is inert)")
print("[dspark-pp3] verified: gate present, refusal retained, module loads")
PY

echo "=== dspark-pp3: done ==="
