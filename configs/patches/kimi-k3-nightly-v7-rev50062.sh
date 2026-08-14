#!/usr/bin/env bash
# v7 with vllm#50062 reverted, to find out whether it is what costs DSpark.
#
# WHAT THE LADDER SHOWED. The 08-14 nightly is neutral for the no-speculation
# arm -- c1 to c78 all inside -1.3% to +1.8% against a 2.0% spread -- and costs
# the DSpark arm 2.2% at c1, 5.2% at c4, 7.3% at c8 and 13.8% at c16. The DSpark
# advantage at c16 fell from +21.7% to +5.3%. GPU prefix hit fell with it, by 3
# to 9 points, on the DSpark arm only.
#
# WHY THIS COMMIT. Of the 59 in the window, f80b66f548 is
# "[Model Runner V2][Spec Decode] Add KV cache support for multi-layer MTP"
# (#50062), and it touches kv_cache_coordinator.py, sched/scheduler.py,
# single_type_kv_cache_manager.py, kv_cache_interface.py and speculative.py.
# We run VLLM_USE_V2_MODEL_RUNNER=1 with speculation, so it is on our path; a
# no-speculation arm is not, which is the shape of the loss.
#
# READING COMMIT TITLES HAS BEEN WRONG THREE TIMES on this track -- #51726,
# #51739 and #51311 were each predicted and each refuted -- so this is a
# measurement, not a conclusion. If reverting it does not recover the DSpark
# arm, the window gets bisected the way the 66-commit one was.
#
# The revert is clean on top of our eight: git auto-merged speculative.py,
# kv_cache_coordinator.py and sched/scheduler.py with no conflicts. Patch is 20
# files against v7's 15, still pure Python.
#
# NOT A SHIPPABLE CONFIGURATION either way. #50062 adds multi-layer MTP KV
# support; dropping it is a way to attribute a number, not a recipe.
set -euo pipefail

# The nightly this patch was generated against: vllm/vllm-openai:nightly-3ee2df3033...
readonly PINNED_SHA=ac7509e2b1
readonly PATCH=/configs/patches/k3-ours8-v7-rev50062.patch

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
print("=== k3-ours8-v7-rev50062 verified (today's nightly, #50062 reverted) ===")
PY

# The AST checks prove the patch is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every one of them and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
