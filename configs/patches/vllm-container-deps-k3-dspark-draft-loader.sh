#!/usr/bin/env bash
# Keep the DSpark draft loader off fastsafetensors under pipeline parallelism.
# =============================================================================
# WHAT BREAKS. On the 2026-09-08 nightly (9ea8f3ff) every one of our speculative
# arms -- c24, c32, c48, with and without int64idx -- dies the same way, ten
# minutes into model load, on both PP nodes:
#
#   [Rank 10] Watchdog caught collective operation timeout:
#   WorkNCCL(SeqNum=497221, OpType=BROADCAST, NumelIn=1, NumelOut=1) ...
#   last enqueued work: 497221, last started work: -1
#
# Only the PP1 ranks (8-15) time out. The PP0 ranks (0-7) merely "observe a dump
# signal from another rank": they never enqueued that broadcast at all. Their
# log shows why -- at 10:09:24 PP0 finished loading the target ("Model loading
# took 98.41 GiB memory and 233.8 seconds") and went straight to "Setting kv
# cache block size". It never loaded a draft model. 10:19:24 - 600 s is 10:09:24
# exactly: PP1 enqueued the broadcast the moment PP0 stopped participating.
#
# The stack trace of the failed collective on rank 10 names the caller:
#
#   fastsafetensors/frameworks/_torch.py:148  broadcast
#   fastsafetensors/tensor_factory.py:143     shuffle
#   fastsafetensors/parallel_loader.py        ...
#   vllm/.../weight_utils.py:1119             fastsafetensors_weights_iterator
#   vllm/models/kimi_k3/nvidia/dspark_mla.py:36  _duplicate_context_kv_weights
#   vllm/v1/worker/gpu/spec_decode/dspark/utils.py:76  load_dspark_model
#   vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:94  load_draft_model
#
# THE CAUSE is a collision between two upstream facts. #50514 (d87a440f88,
# "Support eagle3 spec decode with pipeline parallel") instantiates the drafter
# only on the last PP stage. And weight_utils.fastsafetensors_weights_iterator
# hard-codes its process group:
#
#   weight_utils.py:1084    pg = torch.distributed.group.WORLD
#
# so the parallel loader's shuffle collectives over all 16 ranks. The draft
# loader inherits load_format=fastsafetensors from the target config (ours sets
# it, and for a 1.4 TB target it is the right choice), the first stage never
# builds a drafter and never joins, and the last stage waits out the NCCL
# timeout. Nothing about int64idx or our recipe is involved.
#
# We have hit this before. Our own dspark-pp carry (k3-dspark-pp-828.patch,
# from local vLLM commit 540423f2c7 "PP에서 드래프트 가중치 로더가 fastsafetensors를
# 쓰지 않게 한다") fixed it by dropping the draft to load_format="auto" when
# pp_size > 1; that is why the 08/28 image has always worked. Retiring the carry
# in favour of upstream's #50514 re-exposed it, because #50514 does not carry
# the fix.
#
# THE FIX is that same block, inserted into upstream's load_dspark_model at the
# point our carry had it. The draft is a handful of layers, so the parallel
# loader buys nothing there anyway. This is a real defect in #50514 and belongs
# upstream; this script is the reproduction-backed carry until it lands.
# =============================================================================
set -euo pipefail

echo "=== dspark-draft-loader: keep the draft off fastsafetensors under PP ==="

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
target = os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")
if not os.path.exists(target):
    sys.exit("[draft-loader] FATAL: no dspark/utils.py in this image")

src = open(target).read()
MARK = "load_config=replace(draft_vllm_config.load_config, load_format=\"auto\")"

if MARK in src:
    print("[draft-loader] already present in this image")
    sys.exit(0)

ANCHOR = (
    "    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)\n"
    "\n"
    "    with set_model_tag(\"dspark_head\"):\n"
)
if src.count(ANCHOR) != 1:
    sys.exit(
        "[draft-loader] FATAL: expected one quant_config -> set_model_tag anchor in "
        "load_dspark_model, found %d -- has upstream restructured it?" % src.count(ANCHOR)
    )

FIXED = (
    "    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)\n"
    "\n"
    "    # Under PP the drafter is built only on the last stage, but the\n"
    "    # fastsafetensors loader collectives over torch.distributed.group.WORLD\n"
    "    # (weight_utils.fastsafetensors_weights_iterator). The stages that never\n"
    "    # build a drafter never join, and every rank hangs until the NCCL collective\n"
    "    # times out. Fall back to the default loader for the draft; it is a handful\n"
    "    # of layers, so the parallel loader buys nothing here anyway.\n"
    "    if get_pp_group().world_size > 1 and draft_vllm_config.load_config.load_format == (\n"
    "        \"fastsafetensors\"\n"
    "    ):\n"
    "        draft_vllm_config = replace(\n"
    "            draft_vllm_config,\n"
    "            load_config=replace(draft_vllm_config.load_config, load_format=\"auto\"),\n"
    "        )\n"
    "\n"
    "    with set_model_tag(\"dspark_head\"):\n"
)
# Decide about the import BEFORE inserting the guard: the guard itself contains
# the token get_pp_group, so testing after the insert would always say "present"
# and ship a NameError. Test for the import line, not the bare name.
IMPORT_LINE = "from vllm.distributed.parallel_state import get_pp_group\n"
need_import = IMPORT_LINE not in src and "import get_pp_group" not in src

src = src.replace(ANCHOR, FIXED, 1)

# `replace` is already imported from vllm.config (line 6 upstream); get_pp_group is not.
IMPORT_ANCHOR = "from vllm.config import ModelConfig, VllmConfig, replace\n"
if need_import:
    if src.count(IMPORT_ANCHOR) != 1:
        sys.exit("[draft-loader] FATAL: cannot find the vllm.config import to anchor get_pp_group")
    src = src.replace(
        IMPORT_ANCHOR,
        IMPORT_ANCHOR + "from vllm.distributed.parallel_state import get_pp_group\n",
        1,
    )

compile(src, target, "exec")
open(target, "w").write(src)
print("[draft-loader] applied: " + target)
PY

# Verify what the arms depend on: the guard is in place, and the loader it guards
# against still collectives over WORLD (if upstream ever narrows that group, this
# carry becomes redundant and should be retired rather than kept by inertia).
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
du = open(os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")).read()
wu = open(os.path.join(root, "vllm/model_executor/model_loader/weight_utils.py")).read()

if 'load_format="auto"' not in du or "get_pp_group().world_size > 1" not in du:
    sys.exit("[draft-loader] FATAL: the PP guard is not in load_dspark_model after writing it")
if "from vllm.distributed.parallel_state import get_pp_group" not in du:
    sys.exit("[draft-loader] FATAL: get_pp_group import missing")
if "pg = torch.distributed.group.WORLD" in wu:
    print("[draft-loader] verified: guard present; fastsafetensors still uses group.WORLD, so it is needed")
else:
    print("[draft-loader] note: fastsafetensors no longer hard-codes group.WORLD -- re-check whether this carry is still needed")
PY

echo "=== dspark-draft-loader: done ==="
