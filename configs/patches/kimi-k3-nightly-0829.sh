#!/usr/bin/env bash
# 0829: two commits on the 08-29 nightly, 6d4562c59b (06:09 UTC).
#
# Named for the nightly, not a version counter: kimi-k3-nightly-v9-rev.sh
# and -fix.sh already exist and are the vllm#51766 revert/fix pair, whose
# arms (mcv9r, mcv9f) are in the measured ladder. A third meaning of "v9"
# would collide with them in every result table.
#
# WHY THIS IS SMALLER THAN v8. The 08-28 arm (v8) carried eight files. Two upstream merges
# landed between the 08-28 and 08-29 nightlies and absorbed six of them:
#
#   vllm#53324  [KV Connector] Support MooncakeStore with hybrid DCP prefix
#               caching. Merged 08-29 02:10 UTC. Replaces our whole Mooncake
#               carry -- connector.py, coordinator.py and worker.py.
#   vllm#54167  [Kimi-K3][Bugfix] Fix low-latency GEMM fallback initialization.
#               Merged 08-28 08:32 UTC, 2h19m after the 08-28 nightly was cut,
#               which is why v8 had to carry it and this does not.
#
# and we dropped two more that were dead in every arm we run: the TokenspeedMLA
# flatten env gate (set by 0 of 92 configs) and simple_kv_offload's
# _group_block_size (0 of 34 mcv* arms use kv-offloading-size; all use Mooncake).
# Neither was reachable, so neither was ever measured. They are recoverable from
# k3-ours6-v8-nightly.patch if a later arm needs them.
#
# OURS IS NOT MERELY REDUNDANT WITH #53324 -- IT IS WORSE. Both fix the same
# defect: a fine-grained hit lands inside a dcp-scaled attention block and every
# load of it is -704. We revalidated the boundary and stepped the hit BACK until
# its key existed, throwing the rest of the hit away. #53324 keeps the hit and
# resolves, per group, the hash boundary the tail block was actually stored
# under. Our coordinator hunk still applies cleanly to this nightly -- no
# textual conflict -- so keeping it would have quietly shortened hits upstream
# can now load. It is dropped deliberately, not because patch refused it.
#
# WHAT REMAINS, AND WHY EACH IS STILL HERE.
#
#   kda_metadata.py    A guard, not a fix. _get_aligned_state_indices_kernel
#                      reads column (seq_lens - 1) // block_size and its mask
#                      bounds only the row, so a block table narrower than
#                      max_num_blocks_per_req reads off the end of the row.
#                      That surfaces as an unrelated assert somewhere else, or
#                      as silently wrong recurrent state. Checked once, memoized.
#
#   sched/scheduler.py Upstream still carries "TODO (davidb): add support for
#                      hybrid memory allocator" on this nightly, and the line
#                      under it unpacks get_block_ids() into one group. Kimi-K3
#                      is hybrid, so that raises ValueError on any KV load
#                      failure. Rare -- an eviction race -- but fatal when hit.
set -euo pipefail

# vllm/vllm-openai:nightly-6d4562c59b97b4e35d459ff9389e71b6fe4995de
readonly PINNED_SHA=6d4562c59b
readonly PATCH=/configs/patches/k3-ours2-0829.patch

export FI_VER=0.6.16.post3
# Our commits are applied below, after this returns, so the deps script must not
# assert them yet -- and its OURS list describes the pre-08-28 shape anyway.
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
    if grep -E '^diff --git a/' "$PATCH" | grep -qv 'a/vllm/'; then
        echo "ERROR: $PATCH touches paths outside vllm/; regenerate it with -- vllm/" >&2
        exit 1
    fi
    patch -p1 --forward -d "$SITE" < "$PATCH"
fi

python3 - <<'PY'
import ast
import inspect
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent

# --- ours: both must be present -------------------------------------------
for rel, mark, who in [
    ("models/kimi_k3/nvidia/kda_metadata.py", "def _check_block_table_width",
     "KDA block-table width guard"),
    ("v1/core/sched/scheduler.py", "req_hybrid_block_ids = {",
     "hybrid-aware kv_load_failure recompute"),
]:
    assert mark in (root / rel).read_text(), f"{who}: missing in {rel} after the patch"

# --- upstream: the six we stopped carrying must be here instead ------------
# If any of these is absent, this is not the 08-29 nightly and the arm would run
# without a fix we believe is present. Naming them separately is the difference
# between "the image is wrong" and four hours of -704.
mc = root / "distributed/kv_transfer/kv_connector/v1/mooncake/store"

conn = (mc / "connector.py").read_text()
fn = next(n for n in ast.walk(ast.parse(conn))
          if isinstance(n, ast.FunctionDef) and n.name == "_validate_kv_cache_config")
dcp_refusals = [ast.unparse(n.test) for n in ast.walk(fn)
                if isinstance(n, ast.If) and "dcp" in ast.unparse(n.test)]
assert not dcp_refusals, (
    f"vllm#53324 is missing: a DCP refusal survives in the connector: {dcp_refusals}"
)
assert "pcp > 1" in conn, "the PCP refusal was dropped; it is not covered by #53324"

assert "def contains(" in (mc / "coordinator.py").read_text(), (
    "vllm#53324's ExternalCachedBlockPool.contains is missing"
)
worker = (mc / "worker.py").read_text()
for mark in ("resolve_dcp_kv_cache_spec", "_tail_key_boundaries"):
    assert mark in worker, f"vllm#53324's {mark} is missing from the Mooncake worker"

# #53324 also widened this to every full-attention group, not just the first.
# The second group is the draft KV under EAGLE/DSpark, so on a spec arm the old
# form left it untruncated. Assert the widened shape, not merely the PR number.
coord_src = (mc / "coordinator.py").read_text()
assert "first_group = self.attention_groups[0]" not in coord_src, (
    "the Mooncake coordinator still truncates only attention_groups[0]; "
    "#53324's every-group form is not in this image"
)

from vllm.models.kimi_k3.nvidia.low_latency_gemm import (  # noqa: E402
    _KimiK3LowLatencyApply,
)

assert "super().__init__()" in inspect.getsource(_KimiK3LowLatencyApply.__init__), (
    "vllm#54167 is missing: _KimiK3LowLatencyApply.__init__ does not call super(), "
    "so KimiK3LowLatencyLinearMethod never gets _gemm_impl and the engine dies in "
    "kda.py during profile_run"
)

# --- ours, removed: these must NOT come back -------------------------------
# Our step-back still applies cleanly to this nightly, so nothing but this check
# would notice it. It shortens hits that #53324 can load.
assert "_exact_partial_hit_key_exists" not in coord_src, (
    "our step-back revalidation is back in the coordinator. It is not a conflict "
    "with #53324 -- it silently shortens hits #53324 resolves correctly."
)
print("=== k3-ours2-0829 verified (2 ours, #53324 + #54167 from upstream) ===")
PY

# The AST checks prove the code is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every check we had and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, against #53324's resolver, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
