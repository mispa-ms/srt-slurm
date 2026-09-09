#!/usr/bin/env bash
# Kimi-K3 B200 AGG on the 2026-09-09 nightly (385dce36) with the two remaining
# stopgaps and nothing else.
# =============================================================================
# The 09/08 audit (B200_PP_DISAGG_HANDOFF.md, "What the 09/08 chain actually
# adds") left two runtime patches that upstream has not merged yet:
#   draft-loader   PP drafter must not load via fastsafetensors  (vllm#54416, Draft)
#   bcfix          gather draft tokens on the main stream         (vllm#55745, Open)
# Everything else is either in the image now (#55924 int64 cast, #55472 draft
# ParallelConfig, #53324, #54167), shown unnecessary (emptycache), third-party
# and unproven (SA's DCP dummy-batch fix, #55212), or logging (dcpdiag). This
# chain carries none of those, so the frontier reproduction on 09/09 is
# "upstream main + two open PRs".
#
# hfshim is environment only (points the HF cache at the staged checkpoint);
# vllm-container-deps.sh installs msgpack for the Mooncake connector and its
# numa-hash patch self-skips on images that already carry it.
# =============================================================================
set -euo pipefail

echo "=== k3-b200-909min: 09/09 nightly + loader guard + bcfix ==="

bash /configs/patches/vllm-container-deps.sh
bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-k3-dspark-draft-loader.sh
bash /configs/patches/vllm-container-deps-k3-pp-broadcast-drafts-fix.sh

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
def rd(p): return open(os.path.join(root, p)).read()
kda = rd("vllm/models/kimi_k3/nvidia/kda.py")
du = "".join(rd("vllm/v1/worker/gpu/spec_decode/dspark/utils.py").split())
mr = rd("vllm/v1/worker/gpu/model_runner.py")
pu = rd("vllm/v1/worker/gpu/pp_utils.py")
problems = []
if "checkpoint_state_indices_ptr + seq_idx).to(tl.int64)" not in kda:
    problems.append("int64 cast (#55924) not in this image -- this chain assumes a nightly >= 2026-09-09")
if "parallel_config=replace(vllm_config.parallel_config,pipeline_parallel_size=1" not in du:
    problems.append("#55472 draft ParallelConfig not in this image -- this chain assumes a nightly >= 2026-09-09")
if "get_pp_group().world_size>1" not in du:
    problems.append("draft-loader guard missing")
if "input_batch.dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens" in mr:
    problems.append("SA dummy-batch fix present -- this chain must not carry it")
i = mr.index("self.speculator.load_model(self.model)")
if "empty_cache" in mr[max(0, i - 600):i]:
    problems.append("emptycache present -- this chain must not carry it")
i = pu.index("def broadcast_drafts"); j = pu.index("def receive", i); body = pu[i:j]
if body.index("send = draft_tokens[input_batch.idx_mapping]") > body.index("with torch.cuda.stream(self.broadcast_stream)"):
    problems.append("bcfix missing: draft gather still inside the side-stream block")
if problems:
    sys.exit("[909min] FATAL: " + "; ".join(problems))
print("[909min] verified: image has #55924 + #55472; chain adds exactly loader guard + bcfix; no dummy fix, no emptycache")
PY

echo "=== k3-b200-909min: done ==="
