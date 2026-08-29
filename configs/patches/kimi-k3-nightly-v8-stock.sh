#!/usr/bin/env bash
# The 08-28 nightly with NO patch of ours, to find out what still needs one.
#
# WHY. Our patch has shrunk from nine commits to six as upstream absorbed
# #50493, DSpark-under-DCP and our own #52419. Reading the remaining six says
# Mooncake still refuses DCP > 1 with hybrid attention upstream:
#
#   connector.py:113
#   if len(kv_cache_config.transfer_groups) > 1 and pcp * dcp > 1:
#       unsupported.append(f"PCP/DCP > 1 ... with hybrid attention")
#   ...
#   raise ValueError(...)
#
# Kimi-K3 is hybrid and we run DCP 8, so this should fail at startup. Reading
# has been wrong four times on this track, so it gets measured before we decide
# whether the submission needs a patch or an image at all.
#
# WHAT THIS ARM IS. Deps, plus exactly one upstream commit and nothing of ours:
# vllm#54167. Without it _KimiK3LowLatencyApply omits super().__init__() and the
# model dies with AttributeError: no attribute '_gemm_impl' -- it merged 08-28
# 08:32 and the image was cut 06:13. Carrying it is what makes this arm answer
# the question asked rather than that unrelated one. Our own markers are still
# asserted ABSENT, so a stale layer cannot make a patched image look stock.
#
# EXPECTED: dies at server start with the ValueError above. If it does not, the
# patch may no longer be required and the submission gets simpler.
set -euo pipefail

readonly PINNED_SHA=6f7df92a8e

export FI_VER=0.6.16.post3
export K3_EXPECT_OURS=0
bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-aggv2.sh"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
readonly GEMMFIX=/configs/patches/k3-gemmfix-54167.patch
echo "=== applying vllm#54167 only ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$GEMMFIX" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$GEMMFIX"
fi

python3 - <<PY
import re

import vllm

v = vllm.__version__
m = re.search(r"\+g([0-9a-f]+)", v)
assert m, f"vllm version {v!r} carries no +g<sha>"
sha = m.group(1)
assert sha.startswith("${PINNED_SHA}") or "${PINNED_SHA}".startswith(sha), (
    f"wrong base image: expected ${PINNED_SHA}, got {sha} ({v})"
)
print(f"=== stock nightly verified: {v} ===")
PY

python3 - <<'PY'
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent

# None of ours may be here. A leftover layer would turn this into a patched run
# reported as a stock one, which is the only way this arm can lie.
for rel, mark, who in [
    ("distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py",
     "_exact_partial_hit_key_exists", "#50359 carry"),
    ("v1/simple_kv_offload/manager.py", "def _group_block_size", "offload helper"),
    ("models/kimi_k3/nvidia/kda_metadata.py", "def _check_block_table_width",
     "KDA width guard"),
    ("v1/attention/backends/mla/tokenspeed_mla.py", "VLLM_TS_MLA_DCP_FLATTEN",
     "flatten flag"),
]:
    assert mark not in (root / rel).read_text(), (
        f"{who} is present: this image is not stock, so the result would not "
        "answer the question"
    )

assert "super().__init__()" in (
    root / "models/kimi_k3/nvidia/low_latency_gemm.py"
).read_text(), "vllm#54167 did not apply; the run would die on _gemm_impl instead"

# And the refusal we expect to hit, quoted so the log says what to look for.
conn = (root / "distributed/kv_transfer/kv_connector/v1/mooncake/store"
        / "connector.py").read_text()
if "with hybrid attention" in conn and "pcp * dcp > 1" in conn:
    print("=== stock: Mooncake still refuses DCP>1 + hybrid; expect startup ValueError ===")
else:
    print("=== stock: the DCP+hybrid refusal is GONE upstream -- this may just work ===")
PY
