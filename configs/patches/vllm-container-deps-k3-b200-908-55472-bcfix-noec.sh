#!/usr/bin/env bash
# A/B twin of vllm-container-deps-k3-b200-908-55472-bcfix.sh WITHOUT emptycache.
# =============================================================================
# The 09/08 audit left exactly one runtime change that is ours and not upstream:
# vllm-container-deps-k3-b200-dcp8-emptycache.sh, a `torch.cuda.empty_cache()`
# before `self.speculator.load_model(self.model)`, derived on an 08/2x image
# where every DSpark ns=7 DCP8 arm OOM'd while building the draft. Whether the
# 09/08 nightly still needs it is untested. This chain is the -bcfix chain with
# that one step removed and everything else identical:
#   828.sh (dcp8-diag -> pr54167 -> int64idx) -> draft-loader -> #55472 -> bcfix
# Serves  -> emptycache is not needed on 09/08; retire it, nothing to upstream.
# OOMs at draft load -> it is needed; then it is ours to file (with this log).
# =============================================================================
set -euo pipefail

echo "=== k3-b200-908-55472-bcfix-noec: the -bcfix chain without emptycache ==="

bash /configs/patches/vllm-container-deps-k3-b200-828.sh
bash /configs/patches/vllm-container-deps-k3-dspark-draft-loader.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pr55472.sh
bash /configs/patches/vllm-container-deps-k3-pp-broadcast-drafts-fix.sh

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
kda = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
du = "".join(open(os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")).read().split())
mr = open(os.path.join(root, "vllm/v1/worker/gpu/model_runner.py")).read()
pu = open(os.path.join(root, "vllm/v1/worker/gpu/pp_utils.py")).read()
if "checkpoint_state_indices_ptr + seq_idx).to(tl.int64)" not in kda:
    sys.exit("[bcfix-noec] FATAL: int64 cast missing")
if "parallel_config=replace(vllm_config.parallel_config,pipeline_parallel_size=1" not in du:
    sys.exit("[bcfix-noec] FATAL: #55472 shape missing")
i = mr.index("self.speculator.load_model(self.model)")
if "empty_cache" in mr[max(0, i - 600):i]:
    sys.exit("[bcfix-noec] FATAL: an empty_cache() precedes the draft load -- emptycache leaked into this chain")
i = pu.index("def broadcast_drafts"); j = pu.index("def receive", i); body = pu[i:j]
if body.index("send = draft_tokens[input_batch.idx_mapping]") > body.index("with torch.cuda.stream(self.broadcast_stream)"):
    sys.exit("[bcfix-noec] FATAL: draft gather still inside the side-stream block")
print("[bcfix-noec] verified: int64 cast, #55472 shape, NO empty_cache before the draft load, bcfix present")
PY

echo "=== k3-b200-908-55472-bcfix-noec: done ==="
