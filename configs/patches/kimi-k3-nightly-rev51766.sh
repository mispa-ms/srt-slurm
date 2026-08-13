#!/usr/bin/env bash
# Confirmation: our eight commits on the regressed nightly, with vllm#51766
# reverted and nothing else changed.
#
# WHAT THE BISECT FOUND. Two published nightlies bracket the regression:
# 65b7662d3f is clean at 13,227 tok/s/GPU at c70, 3ee2df3033 shows 11,862.
# A coarse round of seven cuts put the loss inside eight commits; a fine round
# of seven put it on one:
#
#   ce07118669  [Bugfix][Core] Preserve Mamba running CoW after external hits (#51766)
#
#   f4 (through the commit before it) 13,326 tok/s/GPU   ITL p90 124.4 ms
#   f5 (through it)                   12,089             ITL p90 134.1 ms
#   every later cut                   11,5xx-12,1xx      plateau
#
# The change is one line in MambaManager, adding the request to
# _allocated_block_reqs on the early-return path taken when the blocks are
# already there and the hit is not partial:
#
#     if num_required_blocks <= len(req_blocks) and not has_partial_hit:
#   +     self._allocated_block_reqs.add(request_id)
#         return []
#
# WHY THIS CONFIGURATION IS THE ONE THAT PAYS. 69 of this model's 93 layers are
# KDA, Mooncake produces external hits continuously, and at c70 the CPU tier
# carries about half of them -- so that early return is the common case here in
# a way it is not elsewhere. It also explains why everything below c48 was
# unaffected: without cache pressure there are few external hits to preserve.
#
# THIS ARM IS THE CONFIRMATION, not the bisect. It runs the REGRESSED nightly
# with only that commit reverted. If it returns to 13,2xx the attribution is
# closed; if it does not, the fine bisect found a correlate rather than a cause.
set -euo pipefail

# The nightly this patch was generated against: vllm/vllm-openai:nightly-3ee2df3033...
readonly PINNED_SHA=3ee2df3033
readonly PATCH=/configs/patches/k3-ours8-rev51766-nightly.patch

export FI_VER=0.6.16.post3
# Our commits are applied below, after this returns, so the deps script must
# not assert them yet.
export K3_EXPECT_OURS=0
bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-aggv2.sh"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

# Assert the base before touching it. vLLM's dev version carries the commit as
# "+g<sha>", e.g. 0.23.1rc1.dev2223+gf4f10fbcc.
python3 - <<PY
import re

import vllm

v = vllm.__version__
m = re.search(r"\+g([0-9a-f]+)", v)
assert m, f"vllm version {v!r} carries no +g<sha>; cannot verify the base image"
sha = m.group(1)
assert sha.startswith("${PINNED_SHA}") or "${PINNED_SHA}".startswith(sha), (
    f"wrong base image: this patch was generated against ${PINNED_SHA} but the "
    f"image is {sha} ({v}). Re-generate the patch against this nightly rather "
    f"than applying it with fuzz."
)
print(f"=== base nightly verified: {v} ===")
PY

echo "=== applying k3-ours8-rev51766-nightly ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH"
fi

python3 - <<'PY'
import ast
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent

CHECKS = [
    ("v1/worker/gpu/cp_utils.py", "def cp_local_slot(", "DSpark under DCP"),
    ("v1/attention/backends/mla/tokenspeed_mla.py", "VLLM_TS_MLA_DCP_FLATTEN",
     "TokenspeedMLA DCP flatten flag"),
    ("v1/core/sched/scheduler.py", "kv_load_failure", "hybrid-aware recompute"),
]
missing = [(rel, who) for rel, mark, who in CHECKS
           if mark not in (root / rel).read_text()]
assert not missing, f"patch applied but markers are missing: {missing}"

# Mooncake under DCP: the refusal must be gone, PCP must still be refused, and
# #50359's retry must be present -- without it a fine-grained hit lands inside a
# dcp-scaled attention block and every load of it is -704.
mc = root / "distributed/kv_transfer/kv_connector/v1/mooncake/store"
conn = (mc / "connector.py").read_text()
fn = next(n for n in ast.walk(ast.parse(conn))
          if isinstance(n, ast.FunctionDef) and n.name == "_validate_kv_cache_config")
dcp_refusals = [ast.unparse(n.test) for n in ast.walk(fn)
                if isinstance(n, ast.If) and "dcp" in ast.unparse(n.test)]
assert not dcp_refusals, f"a DCP refusal survives in the connector: {dcp_refusals}"
assert "pcp > 1" in conn, "the PCP refusal was dropped; it is not covered by this change"
assert "_exact_partial_hit_key_exists" in (mc / "coordinator.py").read_text(), (
    "vllm#50359's exact-boundary retry is missing"
)

# Upstream's own uniform-decode fix must be here, and ours must not be on top of
# it -- #51865 takes has_prefill where our dropped carry took the token arrays.
import inspect  # noqa: E402

from vllm.v1.worker.utils import get_uniform_decode_token_count  # noqa: E402

params = list(inspect.signature(get_uniform_decode_token_count).parameters)
assert params[-1] == "has_prefill", (
    f"get_uniform_decode_token_count takes {params}, not upstream #51865's form"
)
# The markers kimi-k3-aggv2.sh skipped via K3_EXPECT_OURS=0, checked here now
# that the patch is on. Same list, same file paths -- if it ever diverges from
# that script's OURS list, one of the two is wrong.
for rel, mark, who in [
    ("v1/core/kv_cache_coordinator.py",
     "dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size", "#50493"),
    ("v1/simple_kv_offload/manager.py", "def _group_block_size", "ours"),
    ("v1/worker/gpu/model_runner.py",
     "kv_shard_count = 1 if isinstance(spec, MambaSpec)", "ours"),
    ("models/kimi_k3/nvidia/kda_metadata.py", "def _check_block_table_width", "ours"),
    ("v1/core/sched/scheduler.py", "req_hybrid_block_ids = {", "ours"),
]:
    assert mark in (root / rel).read_text(), f"{who}: missing in {rel} after the patch"
# The revert itself: the early-return path must not register the request.
import ast  # noqa: E402

mgr = (root / "v1/core/single_type_kv_cache_manager.py").read_text()
cls = next(n for n in ast.walk(ast.parse(mgr))
           if isinstance(n, ast.ClassDef) and n.name == "MambaManager")
early = [n for n in ast.walk(cls)
         if isinstance(n, ast.If)
         and "num_required_blocks <= len(req_blocks)" in ast.unparse(n.test)]
assert early, "the MambaManager early-return branch is gone; this revert no longer applies"
body = "\n".join(ast.unparse(s) for s in early[0].body)
assert "_allocated_block_reqs.add" not in body, (
    "vllm#51766's line is still on the early-return path -- the revert did not "
    "take, so this arm is the regressed build and proves nothing"
)
print("=== k3-ours8-rev51766 verified: #51766 reverted ===")
PY

# The AST checks prove the patch is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every one of them and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
