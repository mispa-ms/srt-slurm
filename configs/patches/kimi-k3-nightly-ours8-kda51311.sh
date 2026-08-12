#!/usr/bin/env bash
# Our eight commits plus vllm#51311, on the same official nightly.
#
# The A/B twin of kimi-k3-nightly-ours8.sh: identical patch plus one merged
# upstream commit, so the only variable is #51311.
#
# WHAT #51311 IS. fe889ac92554, merged 2026-08-12 18:51 UTC -- seven hours after
# the nightly this runs on was cut, and after every image on this track, so no
# arm measured so far contains it. It moves the FlashKDA prefill kernel's out,
# final_state and workspace tensors from a per-call torch.empty to buffers the
# workspace manager preallocates, and reports 1.1-1.4x on the kernel.
#
# WHY IT MIGHT NOT BE FREE HERE, WHICH IS THE POINT OF THE A/B. The preallocation
# is sized for the worst case the scheduler admits:
#
#     T = scheduler_config.max_num_batched_tokens   # 16384 on this ladder
#     N = scheduler_config.max_num_seqs             # 2 x concurrency
#     specs = ((1,T,H,D) bf16, (N,H,D,D) fp32, (workspace_size,) uint8)
#
# The (N,H,D,D) fp32 slab is the large one and it is now resident rather than
# transient. Above c48 this ladder is KV-bound -- raising the prefill budget from
# 8192 to 16384 was worth +25.5% at c78 -- so a faster prefill kernel that costs
# KV blocks can lose. The four concurrencies chosen are where that trade would
# show: c48 and c56 on the shoulder, c70 at the peak, c78 past the knee.
#
# 69 of this model's 93 layers are KDA and the workload's ISL is ~115k, so the
# prefill side of the trade is not small either. Which way it goes is measured,
# not assumed.

set -euo pipefail

# The nightly this patch was generated against: vllm/vllm-openai:nightly-3ee2df3033...
readonly PINNED_SHA=3ee2df3033
readonly PATCH=/configs/patches/k3-ours8-kda51311-nightly.patch

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

echo "=== applying k3-ours8-kda51311-nightly ==="
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
# #51311 itself: the kernel now takes preallocated buffers instead of
# allocating them per call. Assert the call site, not the timing.
kda = (root / "models/kimi_k3/nvidia/kda.py").read_text()
assert "current_workspace_manager" in kda, (
    "vllm#51311 is missing: the FlashKDA prefill still allocates per call, so "
    "this arm is not the A/B it claims to be"
)
assert "_flashkda_buffer_specs" in kda, "vllm#51311's buffer specs are absent"
print("=== k3-ours8-kda51311-nightly verified (incl. #51311) ===")
PY

# The AST checks prove the patch is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every one of them and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
