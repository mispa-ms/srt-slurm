#!/usr/bin/env bash
# Carry the target's DCP size into the DSpark draft's parallel config.
# =============================================================================
# WHAT BREAKS. With the fastsafetensors guard in place, the 09/08 arms get past
# model load on both PP stages and then PP1 -- and only PP1 -- dies six seconds
# later, all eight ranks, during profiling KV-cache init:
#
#   _init_minimal_kv_cache_for_profiling -> initialize_kv_cache(is_profiling=True)
#   -> init_attn_backend -> create_metadata_builders -> tokenspeed_mla.py:72
#   -> mla_attention.py:2356  self.dcp_manager = getattr(attention_layer, "dcp_manager", None)
#      mla_attention.py:2357  assert isinstance(self.dcp_manager, MLADCPManager)
#   AssertionError
#
# The metadata builder reads its DCP size from the process group
# (get_dcp_group().world_size == 8) and therefore demands an MLADCPManager on the
# layer. The layer creates that manager only if ITS impl sees DCP > 1, and the
# impl reads DCP from the *config*:
#
#   mla_attention.py:3213  self.dcp_world_size: int = parallel_config.decode_context_parallel_size
#
# The layer that fails is the DRAFT's (only PP1 has a drafter since #50514, which
# is why only PP1 dies), and the draft's ParallelConfig has DCP = 1.
#
# WHY NOW. Upstream 6f7df92a (08/28) built the draft as
#   replace(vllm_config, attention_config=...)
# -- no parallel_config override -- so the draft inherited the target's config,
# DCP = 8 included, and the manager existed. #50514 (d87a440f88) changed one line:
#   + parallel_config=speculative_config.draft_parallel_config,
# and create_draft_parallel_config builds
#   ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=..., <5 copied fields>)
# with no decode_context_parallel_size, so it defaults to 1. Nothing else about
# the draft changed between the two images. #50514 was validated at TP8 x PP2
# without DCP, which is exactly the path that does not reach this assert.
#
# THE FIX restores the 08/28 behaviour in the narrowest way: keep #50514's
# draft_parallel_config (its PP=1 is what puts the drafter on one stage) but carry
# the target's decode_context_parallel_size into it. ParallelConfig declares the
# field as Field(default=1, ge=1) with no coupling to TP/PP in validation, and
# `replace` is already imported at the top of dspark/utils.py. Only dspark/utils.py
# passes draft_parallel_config this way; eagle and dflash do not, so the carry is
# scoped to the one loader that regressed.
#
# Authorship check: an upstream search for this defect was inconclusive (GitHub
# API rate-limited; HTML search returned nothing parseable). Treat "no upstream
# PR" as UNVERIFIED -- re-check before posting anything. This script is a
# stopgap on our branch either way.
# =============================================================================
set -euo pipefail

echo "=== dspark-draft-dcp: carry the target's DCP into the draft parallel config ==="

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
target = os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")
if not os.path.exists(target):
    sys.exit("[draft-dcp] FATAL: no dspark/utils.py in this image")

src = open(target).read()
MARK = "decode_context_parallel_size=vllm_config.parallel_config.decode_context_parallel_size"

if MARK in src:
    print("[draft-dcp] already present in this image")
    sys.exit(0)

ANCHOR = "        parallel_config=speculative_config.draft_parallel_config,\n"
if src.count(ANCHOR) != 1:
    sys.exit(
        "[draft-dcp] FATAL: expected one `parallel_config=speculative_config."
        "draft_parallel_config,` in load_dspark_model, found %d -- if 0, this image "
        "predates #50514 and does not need this" % src.count(ANCHOR)
    )

FIXED = (
    "        # #50514 gives the draft create_draft_parallel_config(), which never sets\n"
    "        # decode_context_parallel_size, so the draft's MLA impl sees DCP=1 and builds\n"
    "        # no MLADCPManager -- while the metadata builder reads the process group\n"
    "        # (DCP=8) and asserts one exists (mla_attention.py:2357). Before #50514 the\n"
    "        # draft inherited the target's parallel config, DCP included. Restore that\n"
    "        # one field.\n"
    "        parallel_config=replace(\n"
    "            speculative_config.draft_parallel_config,\n"
    "            decode_context_parallel_size=(\n"
    "                vllm_config.parallel_config.decode_context_parallel_size\n"
    "            ),\n"
    "        ),\n"
)
src = src.replace(ANCHOR, FIXED, 1)

if "from vllm.config import" not in src or " replace" not in src.split("from vllm.config import", 1)[1].split("\n", 1)[0]:
    sys.exit("[draft-dcp] FATAL: `replace` is not imported from vllm.config in this file")

compile(src, target, "exec")
open(target, "w").write(src)
print("[draft-dcp] applied: " + target)
PY

# Verify what the arm depends on, and that the premise still holds. If the impl
# ever stops reading DCP from the config, or create_draft_parallel_config starts
# carrying it, this carry is redundant and should be retired rather than kept.
python3 - <<'PY'
import ast
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
du_path = os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")
du = open(du_path).read()
mla = open(os.path.join(root, "vllm/model_executor/layers/attention/mla_attention.py")).read()
spec = open(os.path.join(root, "vllm/config/speculative.py")).read()

if "decode_context_parallel_size=vllm_config.parallel_config.decode_context_parallel_size" not in du.replace("\n", "").replace(" ", "").replace(
    "decode_context_parallel_size=vllm_config.parallel_config.decode_context_parallel_size", "decode_context_parallel_size=vllm_config.parallel_config.decode_context_parallel_size"):
    pass  # whitespace-insensitive check below
flat = "".join(du.split())
if "decode_context_parallel_size=(vllm_config.parallel_config.decode_context_parallel_size)" not in flat:
    sys.exit("[draft-dcp] FATAL: the DCP carry is not in load_dspark_model after writing it")

tree = ast.parse(du)
bound = set()
for n in ast.walk(tree):
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            bound.add(a.asname or a.name.split(".")[0])
if "replace" not in bound:
    sys.exit("[draft-dcp] FATAL: `replace` is used but not bound in dspark/utils.py")

if "self.dcp_world_size: int = parallel_config.decode_context_parallel_size" not in mla:
    print("[draft-dcp] note: the MLA impl no longer reads DCP from parallel_config -- re-check whether this carry is still needed")
if "decode_context_parallel_size" in spec.split("def create_draft_parallel_config", 1)[-1][:1500]:
    print("[draft-dcp] note: create_draft_parallel_config now carries DCP itself -- this carry is redundant")
print("[draft-dcp] verified: draft parallel config carries the target's DCP; `replace` bound")
PY

echo "=== dspark-draft-dcp: done ==="
