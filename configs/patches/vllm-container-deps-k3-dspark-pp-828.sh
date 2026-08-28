#!/usr/bin/env bash
# Spec-decode-under-PP, rebased onto the 2026-08-28 nightly (6f7df92a).
# =============================================================================
# WHY IT HAS TO BE PORTED AT ALL. The nightly still refuses the configuration
# outright (model_runner.py):
#
#     if self.use_pp:
#         raise ValueError(f"{method} with pipeline parallel is not supported.")
#
# Six of our seven B200 frontier points are DSpark, so without this the server does
# not start and the frontier cannot move off the pinned image. Nobody upstream runs
# K3 with PP, so nobody upstream is fixing it.
#
# WHAT CHANGED IN THE PORT. k3-dspark-pp-xin.patch was cut against xinli-sw/vllm at
# 728d3ad. Against this nightly, 15 of its 20 hunks still apply and five do not --
# all of them context drift, none of them semantic:
#
#   model_runner.py x2   upstream added `with use_workspace_lane(...)` around the
#                        speculator load and re-indented beneath it
#   kimi_k3/model.py     make_empty_intermediate_tensors moved
#   dspark/utils.py x2   the import block was reorganised, and the _should_share
#                        call grew into a compound condition
#
# The five were rebased by hand against a9a17e7095 and regenerated as a diff. Carried
# forward to 6f7df92a, that version needed exactly one more: 08/28 added
# "fused_qkv_a_g_proj" to packed_modules_mapping, which displaced the class attribute
# this patch inserts beneath it. Rebased and regenerated again, so what is applied here
# is one patch keyed to this image: 7 files, 596 lines, zero failed hunks at fuzz 0,
# round-tripped against a pristine 08/28 tree.
#
# WHAT IT CARRIES, in one line each:
#   - lifts the blanket PP+spec refusal, and replaces it with one that fires only if
#     the model cannot forward aux hidden states across stages
#   - refuses pp > 2 explicitly: the accounting is unit-tested to pp=8 but a *middle*
#     stage has never run on hardware, and this feature fails as degraded acceptance
#     rather than a crash
#   - has the target forward its aux taps alongside IntermediateTensors, and sizes the
#     recv buffers to match
#   - loads the real embedding table on the last stage, where the target's embed is a
#     PPMissingLayer whose forward would hand raw int64 ids to the draft backbone
#
# WHAT IT IS NOT. Not a perf patch. It makes DSpark+PP possible; it does not make it
# fast, and the nightly is currently 10.9% behind the pinned image at c48 no-spec with
# no speculation involved at all.
#
# Verified against .vllm-wt-rubin: applies at fuzz 0, every touched file compiles, and
# each new call site has its symbol present in that module. The container's --dry-run
# below is the real gate and fails the job rather than serving half-patched.
# =============================================================================
set -euo pipefail

echo "=== dspark-pp-828: spec-decode-under-PP for 6f7df92a ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')

if grep -q "The drafter is instantiated only on the last pipeline stage" \
     "$VLLM_ROOT/vllm/config/speculative.py"; then
    echo "[dspark-pp-828] already present in this image"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-dspark-pp-828.patch > /tmp/dspp-dry.log 2>&1; then
        echo "[dspark-pp-828] FATAL: does not apply to this image" >&2
        cat /tmp/dspp-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-dspark-pp-828.patch
    echo "[dspark-pp-828] applied under $VLLM_ROOT"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
rd = lambda p: open(os.path.join(root, p)).read()

mr = rd("vllm/v1/worker/gpu/model_runner.py")
md = rd("vllm/models/kimi_k3/nvidia/model.py")
du = rd("vllm/v1/worker/gpu/spec_decode/dspark/utils.py")

# The refusal being gone is the whole point; a patch that "applied" without removing
# it would fail at server start with a message that reads like a config error.
if 'f"{self.speculative_config.method} with pipeline parallel "\n                        "is not supported."' in mr:
    sys.exit("[dspark-pp-828] FATAL: the blanket PP+spec refusal is still there")

need = [
    (mr, "supports_aux_hidden_states_over_pp(self.model)", "aux-over-PP guard"),
    (mr, "pipeline_parallel_size=2", "pp>2 refusal"),
    (md, "make_empty_aux_hidden_states", "aux recv buffers"),
    (du, "isinstance(target_embed, PPMissingLayer)", "PP embed path"),
    (du, "_load_target_embed_tokens_for_pp", "the loader it calls"),
]
for src, tok, what in need:
    if tok not in src:
        sys.exit("[dspark-pp-828] FATAL: %s missing (%s)" % (what, tok))

import vllm.v1.worker.gpu.spec_decode.dspark.utils  # noqa: F401
import vllm.models.kimi_k3.nvidia.model  # noqa: F401
from vllm.model_executor.models.interfaces import (  # noqa: F401
    supports_aux_hidden_states_over_pp,
)

print("[dspark-pp-828] verified: refusal gone, five call sites present, modules load")
PY

echo "=== dspark-pp-828: done ==="
