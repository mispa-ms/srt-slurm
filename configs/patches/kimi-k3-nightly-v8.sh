#!/usr/bin/env bash
# v8: our commits on the 08-28 nightly, 6f7df92a8e, plus vllm#54167.
#
# The nightly is unusable on this model without #54167: _KimiK3LowLatencyApply
# omits super().__init__(), so KimiK3LowLatencyLinearMethod never gets its
# _gemm_impl and the run dies with AttributeError. It merged 08-28 08:32 and the
# image was cut 06:13 -- two hours short. Carried here as a seventh commit; drop
# it the moment a nightly contains it.
#
# 538 upstream commits on from v7's base, and the patch SHRANK from eight
# commits to six. Three were absorbed:
#
#   #50493 carry     -- upstream now sets enable_partial_hash_hits from
#                       has_partial_mamba_group with no dcp_world_size == 1
#                       gate, which is exactly what our carry did.
#   DSpark under DCP -- landed upstream and extended: null-block guards on the
#                       context and query slots, sliding-window eviction, and
#                       dcp_local_seq_lens passed in rather than recomputed.
#   #52419           -- ours, merged 08-16.
#
# The model-runner half of the CPU-offload commit also went: upstream computes
# the Mamba-aware block count inside spec.max_num_blocks_per_req() instead. The
# marker list below asserts that replacement, because losing it silently
# DCP-scales the Mamba block table again -- a wrong-state bug, not a slow one.
#
# WHAT WE STILL CARRY. Mooncake under DCP with hybrid attention is still refused
# upstream (transfer_groups > 1 and pcp * dcp > 1), so the connector change and
# the per-group block-size scaling in the worker are still ours; the resolution
# keeps our semantics on upstream's newer transfer_groups API and leaves PCP
# refused. Plus the KDA block-table width guard, the hybrid KV-load-failure
# recompute, the TokenspeedMLA flatten flag, and the two #50359 guards.
set -euo pipefail

# The nightly this patch was generated against: vllm/vllm-openai:nightly-3ee2df3033...
readonly PINNED_SHA=6f7df92a8e
readonly PATCH=/configs/patches/k3-ours6-v8-nightly.patch

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
    ("models/kimi_k3/nvidia/low_latency_gemm.py", "super().__init__()",
     "vllm#54167, without which there is no _gemm_impl"),
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
    # Was ours; upstream now does it inside the spec, so assert the replacement
    # rather than our removed line. Losing this silently would DCP-scale the
    # Mamba block table again, which is a wrong-state bug, not a slow one.
    ("v1/worker/gpu/model_runner.py",
     "spec.max_num_blocks_per_req(", "upstream's Mamba-aware block count"),
    ("models/kimi_k3/nvidia/kda_metadata.py", "def _check_block_table_width", "ours"),
    ("v1/core/sched/scheduler.py", "req_hybrid_block_ids = {", "ours"),
]:
    assert mark in (root / rel).read_text(), f"{who}: missing in {rel} after the patch"
# vllm#52419, ours, merged upstream 08-16. If a nightly ever loses it the DSpark
# arm quietly drops ~14% at c16, so check rather than assume.
assert "def _align_cacheable(" in (root / "v1/core/kv_cache_coordinator.py").read_text(), (
    "vllm#52419's _align_cacheable is gone; the EAGLE cache-registration bound "
    "would be re-floored and DSpark loses ~14% at c16"
)

# And must NOT be present here: this is the control.
mgr = (root / "v1/core/single_type_kv_cache_manager.py").read_text()
assert 'self._partial_hit_reqs[request_id] = (block_idx, req_blocks[block_idx])' not in mgr, (
    "fa92f83038 leaked into the control arm; the A/B would compare it to itself"
)
print("=== k3-ours6-v8 verified (08-28 nightly, six commits) ===")
PY

# The AST checks prove the patch is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every one of them and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
