#!/usr/bin/env bash
# Our eight commits on an official vLLM nightly, applied at runtime.
#
# WHY THIS EXISTS. Every arm on this track so far has run an image we built
# ourselves -- k3-dcp-agg-v2, k3-merged-v2/v3/v4. That costs ~50 minutes per
# upstream bump and produces an image nobody else can pull. It is also not what
# SA accepts: the submission path wants an official vLLM image. All eight of our
# commits are pure Python -- 15 files, +356/-62, no .cu, no .cpp, no CMake -- so
# they can ride on the official nightly instead of forcing a rebuild.
#
# The compiled parts of the delta are upstream's own (the Kimi-K3 ROCm KDA decode
# kernel, moe_align_sum, CMakeLists), and a nightly built from main already has
# them. That is the whole reason this works.
#
# WHAT IS IN THE PATCH. Six changes of ours and two carries of still-open PRs:
#   vllm#50493 (carry)  partial prefix-cache hits under DCP
#   ours                hybrid CPU-offload block accounting under DCP, and stop
#                       DCP-scaling the Mamba block table
#   ours                hybrid-aware KV-load-failure recompute in the scheduler
#   ours                DSpark under DCP
#   ours                TokenspeedMLA DCP causal-bound flatten, behind a flag
#   ours                Mooncake store: DCP with hybrid attention
#   vllm#50359 (carry)  revalidate exact partial-hash hit boundaries
#   ours                two guards on #50359's retry that it is missing
#
# WHAT IS NOT IN IT, and why that is fine. Our #50532 carry is dropped: upstream
# landed its own version as 0a94d85a66 (#51865) and this nightly has it. Our
# #50484 cherry-pick went the same way at v4.
#
# WHY THE NIGHTLY IS PINNED TO A SHA rather than the rolling `nightly` tag.
# The patch is generated against one tree. Point this at a moving tag and one
# day it applies to something else -- or worse, applies with fuzz and produces a
# server that runs code nobody reviewed. The pin is asserted below against the
# image's own version string before anything is applied.
#
# TWO MAIN-ONLY COMMITS ARE ABSENT FROM THIS NIGHTLY, and neither costs us
# anything. #51843 adds a manager-capability guard to fine-grained prefix-cache
# hits; both FullAttentionManager and MambaManager declare
# supports_fine_grained_hash_lookup, so it is a no-op on this model's 24 MLA +
# 69 KDA layout. #51917 is a refactor of the uniform-decode helper -- the fix
# itself, #51865, is already here.
set -euo pipefail

# The nightly this patch was generated against: vllm/vllm-openai:nightly-3ee2df3033...
readonly PINNED_SHA=3ee2df3033
readonly PATCH=/configs/patches/k3-ours8-nightly.patch

export FI_VER=0.6.16.post3
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

echo "=== applying k3-ours8-nightly ==="
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
print("=== k3-ours8-nightly verified ===")
PY

# The AST checks prove the patch is present, not that it is right. Lifting the
# Mooncake DCP refusal the first time passed every one of them and then
# livelocked with 2,757,664 OBJECT_NOT_FOUND (-704) failures. This asserts the
# property that failure violates, before any GPU time.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
