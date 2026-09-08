#!/usr/bin/env bash
# The 09/08 chain with upstream #55472 in place of our draft-dcp stopgap.
# =============================================================================
# Identical to vllm-container-deps-k3-b200-908.sh except for the last step: the
# second wall (the draft's ParallelConfig losing DCP under #50514) is closed by
# applying vllm#55472 verbatim rather than by our replace()-DCP-back-in stopgap.
# The two rewrite the same `parallel_config=` block and conflict, so this chain
# carries #55472 alone. Purpose: produce the sentence a review on #55472 needs --
# "applied as-is on 9ea8f3ff, B200 TP8xPP2xDCP8, PP1 passes profiling KV init".
#
# Everything else is unchanged: #53324 / #54167 skip themselves as absorbed,
# int64idx still applies, emptycache still applies, and the fastsafetensors
# draft-loader guard (#54416's mechanism) must precede #55472.
# =============================================================================
set -euo pipefail

echo "=== k3-b200-908-55472: 09/08 chain with upstream #55472 ==="

bash /configs/patches/vllm-container-deps-k3-b200-828.sh
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh
bash /configs/patches/vllm-container-deps-k3-dspark-draft-loader.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pr55472.sh

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
kda = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
du = "".join(open(os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")).read().split())
if "checkpoint_state_indices_ptr + seq_idx).to(tl.int64)" not in kda:
    sys.exit("[908-55472] FATAL: int64 cast missing")
if "parallel_config=replace(vllm_config.parallel_config,pipeline_parallel_size=1" not in du:
    sys.exit("[908-55472] FATAL: #55472 shape missing")
if "decode_context_parallel_size=(vllm_config.parallel_config.decode_context_parallel_size)" in du:
    sys.exit("[908-55472] FATAL: draft-dcp stopgap also present -- chain is mixed")
print("[908-55472] verified: int64 cast, #55472 shape, no stopgap")
PY

echo "=== k3-b200-908-55472: done ==="
