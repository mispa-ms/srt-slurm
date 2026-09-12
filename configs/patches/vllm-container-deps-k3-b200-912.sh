#!/usr/bin/env bash
# Kimi-K3 B200 AGG on the 2026-09-12-or-newer nightly: the loader guard, and nothing else.
# =============================================================================
# By 2026-09-12 every fix this workstream carried is upstream:
#   #55924 int64 checkpoint index      merged 09-08 (bfb443a6b6)
#   #55472 draft ParallelConfig/DCP    merged 09-08 (c268198715)
#   #55745 PP draft-broadcast race     merged 09-09 (e8064a96d0)
#   #55499 TRTLLM ragged prefill sync  merged 09-09 (8c87c333b8) + FlashInfer 0.6.18.post1
#   #54416 fastsafetensors PP draft    merged 09-12 06:22 UTC (dca96bf97b)
# The 09/12 nightly tag was cut at 06:16 UTC, six minutes before #54416 landed, so on
# that image the draft-loader guard is still needed; on 09/13 and later it self-skips
# (it detects upstream's `get_pp_safe_draft_load_config`). Everything else is refused
# here rather than carried: emptycache was A/B'd and found unnecessary, SA's DCP
# dummy-batch fix likewise, and the int64/#55472/bcfix scripts have nothing to do.
# =============================================================================
set -euo pipefail

echo "=== k3-b200-912: latest nightly + the loader guard only ==="

bash /configs/patches/vllm-container-deps.sh
bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-k3-dspark-draft-loader.sh

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
def rd(p):
    return open(os.path.join(root, p)).read()

kda = rd("vllm/models/kimi_k3/nvidia/kda.py")
du = rd("vllm/v1/worker/gpu/spec_decode/dspark/utils.py")
duf = "".join(du.split())
pu = rd("vllm/v1/worker/gpu/pp_utils.py")
tr = rd("vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py")
mr = rd("vllm/v1/worker/gpu/model_runner.py")

missing = []
if "checkpoint_state_indices_ptr + seq_idx).to(tl.int64)" not in kda:
    missing.append("#55924 int64 checkpoint index")
if "parallel_config=replace(vllm_config.parallel_config,pipeline_parallel_size=1" not in duf:
    missing.append("#55472 draft parallel config")
i = pu.index("def broadcast_drafts"); j = pu.index("def receive", i)
if "record_stream" not in pu[i:j]:
    missing.append("#55745 draft-broadcast race fix")
if "skip_all_rows_active_check=True" not in tr:
    missing.append("#55499 TRTLLM ragged prefill fix")
if missing:
    sys.exit("[912] FATAL: this image predates " + ", ".join(missing) + " -- use a newer nightly or a chain that carries them")

carried = []
if "input_batch.dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens" in mr:
    carried.append("SA dummy-batch fix")
k = mr.index("self.speculator.load_model(self.model)")
if "empty_cache" in mr[max(0, k - 600):k]:
    carried.append("emptycache")
if carried:
    sys.exit("[912] FATAL: this chain must carry nothing beyond the loader guard, but found: " + ", ".join(carried))

guard = "get_pp_safe_draft_load_config" in du or "get_pp_group().world_size > 1" in du
print("[912] verified: image carries #55924/#55472/#55745/#55499; draft loader guarded (%s); nothing else carried"
      % ("upstream #54416" if "get_pp_safe_draft_load_config" in du else "our stopgap"))
if not guard:
    sys.exit("[912] FATAL: no PP draft-loader guard present")
PY

echo "=== k3-b200-912: done ==="
