#!/usr/bin/env bash
# Reproduce the v3-mc ladder on an official vLLM nightly, no image build.
#
# WHY THIS PARTICULAR NIGHTLY. v3-mc was built on upstream 0e2d78028c. Docker Hub
# keeps eleven nightly-<sha> tags, and only two bracket that base:
#
#   nightly-b22afe45ac (08-10)  25 commits BEHIND  -- unusable
#   nightly-65b7662d3f (08-11)  35 ahead, 0 behind -- this one
#
# The 08-10 nightly is numerically closer but it is missing #49436, whose
# _memcpy_u64_tiled in vllm/v1/worker/mamba_utils.py is the marker that
# distinguishes v3 from v2 in the first place, and #50484. Reproducing v3 on an
# image that predates v3's own content is not reproducing it. 65b7662d3f is the
# earliest published nightly that contains everything v3-mc's base did.
#
# NINE COMMITS, NOT TEN. git dropped our #50484 carry during the rebase --
# "patch contents already upstream" -- because 63ac04a61e is in this nightly.
# That is the same reason v4's branch lost it.
#
# WHAT THIS IS NOT. It is not the artifact the published v3-mc numbers were
# measured on. Those ran image k3-merged-v3 (seven commits) plus the three
# Mooncake commits as a runtime patch, on upstream 0e2d78028c exactly. This runs
# the same nine changes on a nightly 35 commits later. If the numbers differ,
# those 35 upstream commits are the difference, and that is worth knowing --
# it is a much shorter list than the 68 between v3 and v4.
#
# The patch is pure Python: 19 files, +477/-77, no .cu, no .cpp, no CMake.
set -euo pipefail

readonly PINNED_SHA=65b7662d3f
readonly PATCH=/configs/patches/k3-v3mc-on-nightly.patch

export FI_VER=0.6.16.post3
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
    f"wrong base image: this patch was generated against ${PINNED_SHA} but the "
    f"image is {sha} ({v})."
)
print(f"=== base nightly verified: {v} ===")
PY

echo "=== applying k3-v3mc-on-nightly ==="
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
    ("v1/worker/utils.py", "def get_uniform_decode_token_count", "#50532 carry"),
    # upstream's, and the reason this nightly was chosen over the 08-10 one
    ("v1/worker/mamba_utils.py", "_memcpy_u64_tiled", "#49436, absent from v2"),
]
missing = [(rel, who) for rel, mark, who in CHECKS
           if mark not in (root / rel).read_text()]
assert not missing, f"patch applied but markers are missing: {missing}"

mc = root / "distributed/kv_transfer/kv_connector/v1/mooncake/store"
conn = (mc / "connector.py").read_text()
fn = next(n for n in ast.walk(ast.parse(conn))
          if isinstance(n, ast.FunctionDef) and n.name == "_validate_kv_cache_config")
dcp_refusals = [ast.unparse(n.test) for n in ast.walk(fn)
                if isinstance(n, ast.If) and "dcp" in ast.unparse(n.test)]
assert not dcp_refusals, f"a DCP refusal survives in the connector: {dcp_refusals}"
assert "pcp > 1" in conn, "the PCP refusal was dropped"
assert "_exact_partial_hit_key_exists" in (mc / "coordinator.py").read_text(), (
    "vllm#50359's exact-boundary retry is missing"
)
print("=== k3-v3mc-on-nightly verified ===")
PY

echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
