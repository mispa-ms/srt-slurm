#!/usr/bin/env bash
# Bisect cut 1 of 7: the clean image with upstream Python up to a311916a29.
#
# One of seven arms submitted together rather than in sequence. Each carries the
# same clean nightly (65b7662d3f, measured at 13,227 tok/s/GPU at c70) with the
# window's Python applied only as far as its own cut point, plus our commits.
# The regressed end is nightly-3ee2df3033 at 11,862. Reading the seven together
# localises the cause to one eighth of the 66-commit window in a single round
# instead of six sequential ones.
#
#   cut 1: a311916a29 08-11 00:06 [MRV2][Spec] Fuse AR speculator multi-step decodes back into one CUDA graph (#46849)
#
# Nothing compiled moves in any arm -- torch_bindings.cpp and ops.h are identical
# across the whole window, so the ABI is fixed and later Python cannot call an op
# the older .so lacks. Three .cu files do change in the window and are
# deliberately excluded; if all seven arms come back clean, they and the
# transformers 5.14.1 -> 5.15.0 bump are what is left.
set -euo pipefail

readonly PINNED_SHA=65b7662d3f
readonly PATCH=/configs/patches/k3-bisect-cut1.patch

export FI_VER=0.6.16.post3
# Our commits arrive inside the patch below, not before it.
export K3_EXPECT_OURS=0
bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-aggv2.sh"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

python3 - <<PY
import re

import vllm

v = vllm.__version__
m = re.search(r"\+g([0-9a-f]+)", v)
assert m, f"vllm version {v!r} carries no +g<sha>; cannot verify the base image"
sha = m.group(1)
assert sha.startswith("${PINNED_SHA}") or "${PINNED_SHA}".startswith(sha), (
    f"wrong base image: this bisect step needs the CLEAN nightly ${PINNED_SHA}, "
    f"got {sha} ({v}). Running it on the later nightly would compare that image "
    f"against itself."
)
print(f"=== clean base nightly verified: {v} ===")
PY

echo "=== applying k3-bisect-cut1 ==="
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

# Our eight, which ride inside the patch.
for rel, mark, who in [
    ("v1/worker/gpu/cp_utils.py", "def cp_local_slot(", "DSpark under DCP"),
    ("v1/attention/backends/mla/tokenspeed_mla.py", "VLLM_TS_MLA_DCP_FLATTEN", "flatten flag"),
    ("v1/core/kv_cache_coordinator.py",
     "dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size", "#50493"),
    ("v1/simple_kv_offload/manager.py", "def _group_block_size", "ours"),
    ("v1/worker/gpu/model_runner.py",
     "kv_shard_count = 1 if isinstance(spec, MambaSpec)", "ours"),
    ("models/kimi_k3/nvidia/kda_metadata.py", "def _check_block_table_width", "ours"),
    ("v1/core/sched/scheduler.py", "req_hybrid_block_ids = {", "ours"),
]:
    assert mark in (root / rel).read_text(), f"{who}: missing in {rel} after the patch"

# And a marker from the LATER window, to prove the forward Python actually landed
# rather than the patch no-oping. #51865 is upstream's uniform-decode fix, merged
# inside the window; its has_prefill signature is absent from the clean image.
# This cut predates #51865, so our own #50532 carry is still the one in play.
# Proof that forward Python landed comes from the cut's own boundary commit
# instead: a311916a29 08-11 00:06 [MRV2][Spec] Fuse AR speculator multi-step deco
import inspect  # noqa: E402

from vllm.v1.worker.utils import get_uniform_decode_token_count  # noqa: E402

params = list(inspect.signature(get_uniform_decode_token_count).parameters)
assert "has_prefill" not in params, (
    f"get_uniform_decode_token_count takes {params}, which is #51865's form -- "
    "this cut is supposed to predate it, so the wrong patch was applied"
)

mc = root / "distributed/kv_transfer/kv_connector/v1/mooncake/store"
conn = (mc / "connector.py").read_text()
fn = next(n for n in ast.walk(ast.parse(conn))
          if isinstance(n, ast.FunctionDef) and n.name == "_validate_kv_cache_config")
assert not [n for n in ast.walk(fn)
            if isinstance(n, ast.If) and "dcp" in ast.unparse(n.test)], (
    "a DCP refusal survives in the Mooncake connector"
)
assert "_exact_partial_hit_key_exists" in (mc / "coordinator.py").read_text()
print("=== k3-bisect-cut1 verified ===")
PY

echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
