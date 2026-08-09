#!/usr/bin/env bash
# Runtime setup for the merged image: one image, both ladders, no source patching.
#
# The AGG v2 and DSpark tracks had been running the same base image plus their
# own runtime patches -- ours vllm-pr50532-uniform-decode.patch, theirs
# k3-dspark-on-aggv2.patch, which carried a byte-identical copy of the same PR.
# mispa-ms/vllm@misunp/k3-merged-v2 carries all of it as commits, so this script
# installs dependencies and then only checks that the image is the one it thinks
# it is.
#
# THAT CHECK IS THE POINT. Runtime patching fails loudly when a hunk stops
# applying, but succeeds silently when the wrong image is pinned and the patch
# was already there -- and an arm that quietly ran a different binary is scored
# as though it did not. With nothing left to patch, the only remaining failure
# mode is the wrong image, so every marker below is checked and the run dies
# rather than produce a number.
#
# Do not add `patch -p1` here. If something new is needed, commit it to the
# branch and rebuild -- the whole point of the merge was to stop the two tracks
# drifting apart one runtime patch at a time.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-aggv2.sh"

python3 - <<'PY'
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent

# (path, marker, which commit put it there)
CHECKS = [
    # vllm#50532 -- uniform-decode dispatch must check request state, not shape
    ("v1/worker/utils.py", "def get_uniform_decode_token_count", "#50532"),
    ("v1/worker/gpu/cudagraph_utils.py", "get_uniform_decode_token_count", "#50532"),
    # DSpark under DCP
    ("v1/worker/gpu/cp_utils.py", "def cp_local_slot(", "dspark"),
    # TokenspeedMLA DCP causal-bound flatten, behind VLLM_TS_MLA_DCP_FLATTEN
    ("v1/attention/backends/mla/tokenspeed_mla.py", "VLLM_TS_MLA_DCP_FLATTEN",
     "flatten flag"),
]

missing = [(rel, who) for rel, mark, who in CHECKS
           if mark not in (root / rel).read_text()]
assert not missing, (
    f"wrong image: expected misunp/k3-merged-v2, missing {missing}. "
    "The AGG v2 image (k3-dcp-agg-v2) carries none of these."
)
print("=== merged image verified: #50532, dspark, flatten flag ===")
PY
