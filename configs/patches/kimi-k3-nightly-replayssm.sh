#!/usr/bin/env bash
# Our eight commits plus ReplaySSM for Kimi-K3 spec decode, on the nightly.
#
# WHAT REPLAYSSM IS FOR. The DSpark cliff on this track is a memory problem, not
# an acceptance problem: the draft model takes 38% of the GPU KV pool, GPU prefix
# hit at c32 falls 92.8% -> 56%, and TTFT p90 goes to 24.7 s while acceptance
# holds at 40-41%. Every step that narrowed the cliff was a memory measure --
# SimpleCPUOffload -45.6%, Mooncake -19.9%, prefill budget 16384 -14.6%.
# ReplaySSM caches SSM inputs and replays accepted state instead of keeping
# per-token state blocks, which is the same lever.
#
# WHICH BRANCH. ZJY0516/vllm@replayssm-k3-mtp, five commits, based on exactly
# this nightly (3ee2df30) so it rebases onto our eight with no conflicts. The
# branch linked in chat, k3/replayssm, is based on 9c110fa5 (08-01) and does not
# apply: 437 commits later, upstream #49436 replaced the state-copy Triton
# kernel with a 3D-tiled version and ReplaySSM's SKIP_TEMPORAL_COPY /
# USE_TEMPORAL_TOKEN_BIAS flags are written against the old signature. Resolving
# that by hand would mean reimplementing kernel semantics we do not own.
#
# WHY SIMPLECPUOFFLOAD AND NOT MOONCAKE. vllm/config/vllm.py refuses
# --use-replayssm when kv_transfer_config.is_kv_transfer_instance, which is
# `kv_connector is not None and kv_role in KVRole`. SimpleCPUOffload sets neither
# -- it is a separate subsystem (vllm/v1/simple_kv_offload/) and its
# kv-transfer-config carries only a failure policy -- so it passes. Mooncake sets
# both and is refused. That guard is upstream's, from #48018.
#
# SO THIS ARM IS NOT OUR DEPLOYED CONFIGURATION. The track's best numbers are on
# Mooncake; this measures ReplaySSM against a SimpleCPUOffload baseline, whose
# cliff is -45.6%. It answers "how much does ReplaySSM recover on its own",
# which is the question the author needs, not "should we ship it".
#
# ACCURACY IS NOT OPTIONAL HERE. ReplaySSM reconstructs SSM state rather than
# storing it. The perf harness runs with ignore_eos and would score a corrupted
# state as a fast run, so the GSM8K arms are part of this experiment, not a
# follow-up.

set -euo pipefail

# The nightly this patch was generated against: vllm/vllm-openai:nightly-3ee2df3033...
readonly PINNED_SHA=3ee2df3033
readonly PATCH=/configs/patches/k3-ours8-replayssm-nightly.patch

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

echo "=== applying k3-ours8-replayssm-nightly ==="
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
# ReplaySSM itself. The config-level switch is --use-replayssm; these assert
# the code that switch reaches, so a patch that applied but landed the wrong
# branch cannot pass.
assert (root / "models/kimi_k3/nvidia/ops/replayssm.py").exists(), (
    "ReplaySSM ops are missing; this is not the replayssm patch"
)
from vllm.config.cache import CacheConfig  # noqa: E402

assert hasattr(CacheConfig, "use_replayssm_spec"), (
    "CacheConfig has no use_replayssm_spec: the K3 spec-decode half of "
    "ReplaySSM is absent"
)
kdam = (root / "models/kimi_k3/nvidia/kda_metadata.py").read_text()
assert "commit_replayssm_state" in kdam, "the KDA replay commit path is missing"
print("=== k3-ours8-replayssm-nightly verified (incl. ReplaySSM) ===")
PY

# The AST checks prove the patch is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every one of them and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
