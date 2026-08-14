#!/usr/bin/env bash
# v7 plus a ninth commit: restore the fine-grained cache-registration bound that
# vllm#50062 undoes.
#
# WHAT THE LADDER FOUND. The 08-14 nightly is neutral for no-speculation -- c1 to
# c78 within -1.3% to +1.8% -- and costs DSpark 7.4% at c8 and 13.8% at c16, with
# GPU prefix hit down 8.5 points on that arm only. Two repeats put the v7 DSpark
# numbers within 0.1-0.3% of each other, so it is not run-to-run spread.
#
# WHAT IT IS. HybridKVCacheCoordinator.cache_blocks computes
# aligned_num_computed_tokens once, and our #50493 carry deliberately leaves it
# UN-floored when enable_partial_hash_hits is on (a hybrid Mamba group, which is
# this model). vllm#50062 rewrote the EAGLE branch to re-derive its own
# alignment from num_finalized_computed_tokens with an unconditional
#     // scheduler_block_size * scheduler_block_size
# so the floor comes back on the one path that had removed it.
#
# The size of it, from this run's own log: TOKENSPEED_MLA sets block_size 32 and
# the attention block is forced to 1536 to cover the mamba page, so the cap is
# floor(n/1536)*1536 + 32 instead of n -- up to 1,504 tokens of every prefix tail
# never registered. Only EAGLE-family groups take the branch, which is exactly
# why no-spec is untouched.
#
# THE FIX is four lines: respect enable_partial_hash_hits when aligning, leaving
# #50062's num_reprefillable_tokens logic alone. Checked arithmetically to be
# identical to the v6 result across block-size combinations.
#
# FOUND BY READING, not by bisecting. The first candidate (#50062 itself, via
# num_prefill_lookahead) was ruled out the same way -- at lookahead 1 every
# scheduler-side consumer degenerates -- and the real interaction is with our own
# patch, which a bisect of upstream commits alone would not have shown.
set -euo pipefail

# The nightly this patch was generated against: vllm/vllm-openai:nightly-3ee2df3033...
readonly PINNED_SHA=ac7509e2b1
readonly PATCH=/configs/patches/k3-ours9-v7-cachefix.patch

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

echo "=== applying $(basename "$PATCH") ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH" >/dev/null 2>&1; then
    echo "already applied"
else
    # The patch must contain only vllm/ paths. site-packages has no tests/ tree,
    # so a tests/ hunk makes patch exit non-zero, and with set -e that kills the
    # setup after the deps have already been installed -- which reads as "server
    # never became healthy" two log screens later. v7 shipped that way once.
    if grep -qE '^diff --git a/(?!vllm/)' "$PATCH" 2>/dev/null || \
       grep -E '^diff --git a/' "$PATCH" | grep -qv 'a/vllm/'; then
        echo "ERROR: $PATCH touches paths outside vllm/; regenerate it with -- vllm/" >&2
        exit 1
    fi
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
    ("v1/core/kv_cache_coordinator.py", "if self.enable_partial_hash_hits:\n                    aligned_num_finalized_computed_tokens",
     "the #50062 alignment fix"),
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
# And must NOT be present here: this is the control.
mgr = (root / "v1/core/single_type_kv_cache_manager.py").read_text()
assert 'self._partial_hit_reqs[request_id] = (block_idx, req_blocks[block_idx])' not in mgr, (
    "fa92f83038 leaked into the control arm; the A/B would compare it to itself"
)
print("=== k3-ours9-v7-cachefix verified ===")
PY

# The AST checks prove the patch is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every one of them and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
