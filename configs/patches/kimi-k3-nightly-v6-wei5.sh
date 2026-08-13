#!/usr/bin/env bash
# v6 + five of Wei's fixes. Treatment arm of the A/B.
#
# Today's nightly, 3d204dfdaa (08-12 21:38 UTC, pushed 08-13). #51766 is still
# in it, so the c48+ regression this track bisected is expected to be present:
# reverting that one line on the 08-12 nightly recovered +8.9% at c70, 77% of
# the 1,365 tok/s/GPU gap.
#
# WHY THESE FIVE. wzhao/kimi-k3-agentx-v2 carries our work plus six fixes of his
# own, none of them upstream. One is next to the commit we bisected:
#
#   fa92f83038  fix(prefix-cache): preserve partial-hit CoW ownership
#     -  self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
#     +  self._partial_hit_reqs[request_id] = (block_idx, req_blocks[block_idx])
#     -  if self._has_partial_local_hit(new_computed_blocks, ...)
#     +  if self._has_partial_local_hit(req_blocks, ...)
#
# That is the state behind `has_partial_hit`, which is half the condition on the
# line we found:
#
#   if num_required_blocks <= len(req_blocks) and not has_partial_hit:
#       self._allocated_block_reqs.add(request_id)     # <- vllm#51766
#       return []
#
# So if the partial-hit test was wrong, how often that line runs was wrong too --
# which is also the open question this track could not answer, since with
# num_speculative_blocks = 0 on a no-spec arm both of its consumers should
# compute the same value either way.
#
# The other four are DCP/NIXL and KV-cache accounting fixes from the same branch.
#
# FIVE, NOT SIX. 28e0a6bd6f (enforce free-block queue invariants) conflicts:
# upstream has since added `and not block.is_null` to the branch it rewrites.
# Resolving someone else's unmerged patch by hand risks changing what it means,
# so it is left out and named here rather than guessed at. Nothing else in the
# series depends on it -- fa92f83038 applies cleanly without it.
set -euo pipefail

# The nightly this patch was generated against: vllm/vllm-openai:nightly-3ee2df3033...
readonly PINNED_SHA=3d204dfdaa
readonly PATCH=/configs/patches/k3-ours8-wei5-v6-nightly.patch

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

echo "=== applying k3-ours8-wei5-v6-nightly ==="
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
# Wei's partial-hit fix must be present, or this arm is the control.
mgr = (root / "v1/core/single_type_kv_cache_manager.py").read_text()
assert 'self._partial_hit_reqs[request_id] = (block_idx, req_blocks[block_idx])' in mgr, (
    "fa92f83038 is missing: _partial_hit_reqs still stores new_computed_blocks[-1], "
    "so this is the control arm rather than the treatment"
)
print("=== k3-ours8-wei5-v6 verified (incl. Wei's partial-hit CoW fix) ===")
PY

# The AST checks prove the patch is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every one of them and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
