#!/usr/bin/env bash
# Fine bisect 1 of 7: one commit added to the last clean point.
#
# The coarse round put the regression inside eight commits. cut1
# (a311916a29, window commit 8) read 13,108 tok/s/GPU at c70 -- clean against the
# 13,227 baseline -- and cut2 (6c95a641e9, commit 16) read 12,068, -8.8%. Every
# later cut sat on that same plateau, so the loss happens once, inside this
# window, and does not accumulate.
#
# These seven arms walk the window one commit at a time on the clean image. The
# eighth point is cut2 itself and is already measured, so the first arm that
# drops names the commit.
#
#   through: 529d010351 08-11 01:17 [Doc] Fix typos in speculative decoding docs (#51500)
#
# Nothing compiled moves; the whole 66-commit window leaves torch_bindings.cpp
# and ops.h untouched, which is why any of this can be done with patches.
set -euo pipefail

readonly PINNED_SHA=65b7662d3f
readonly PATCH=/configs/patches/k3-bisect-f1.patch

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

echo "=== applying k3-bisect-f1 ==="
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
print("=== k3-bisect-f1 verified ===")
PY

echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
