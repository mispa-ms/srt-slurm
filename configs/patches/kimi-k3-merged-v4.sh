#!/usr/bin/env bash
# Runtime setup for k3-merged-v4: our nine commits on 2026-08-11 upstream.
#
# v4 is v3-mc's branch rebased from 0e2d78028c onto e3fe212eaf, 68 commits later,
# and it exists to answer the same question v3 did -- what does moving to current
# main cost or buy -- for a main that has since absorbed work aimed squarely at
# this model:
#   #50484 landed upstream as 63ac04a61e, so our cherry-pick of it is gone from
#     the branch. Nine commits here, not ten.
#   #46849 fuses the AR speculator's multi-step decodes back into one CUDA graph.
#     That is the DSpark arms' inner loop, and it rewrote the same file our
#     #50532 carry touches -- the conflict was resolved by keeping upstream's
#     BlockTables/AttentionGroup imports and applying only the
#     get_uniform_token_count -> get_uniform_decode_token_count rename.
#   #51766 preserves Mamba running CoW after external hits, which on a 24-MLA +
#     69-KDA model is exactly the Mooncake read path.
#   #51749 generalises KV block zeroing to AttentionSpec.
# So a v3->v4 delta on the DSpark and Mooncake arms is not a rebase formality;
# those three are the reason for the build.
#
# FLASHINFER IS UNCHANGED. requirements/ moves only transformers 5.14.1 ->
# 5.15.0 across those 68 commits, so FI stays at v3's 0.6.16.post3. It is still
# pinned explicitly rather than left to the image, so the version lands in the
# log next to the numbers.
#
# WHY THE EXTRA MARKER. kimi-k3-merged-v3.sh checks five markers that v3 and v4
# both carry, so on its own it cannot tell the two images apart -- mispin v3 here
# and the run would succeed and be scored as a v4 result, which is the whole
# comparison. _fused_multi_step_decode arrives with #46849 and is absent from
# v3, so it separates them, and it is the right marker to pick because it is
# also the change most likely to move the DSpark numbers.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-merged-v3.sh"

python3 - <<'PY'
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent
rel = "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
mark = "_fused_multi_step_decode"
path = root / rel
assert path.exists() and mark in path.read_text(), (
    f"wrong image: expected k3-merged-v4, but {rel} does not contain {mark!r}. "
    "That method arrives with vllm#46849 (2026-08-11) and is absent from "
    "k3-merged-v3, so this is a v3 image."
)

# #46849 rewrote the file our #50532 carry edits. v3's marker check runs before
# this one and would catch an outright loss, but the two changes meet inside
# the same dispatch, so assert they coexist rather than trusting the order the
# rebase happened to produce.
src = path.read_text()
assert "get_uniform_decode_token_count" in src or "get_uniform_decode_token_count" in (
    root / "v1/worker/utils.py").read_text(), (
    "vllm#50532's uniform-decode dispatch did not survive the #46849 rebase"
)
print("=== v4 image verified: #46849 fused multi-step decode + #50532 ===")
PY
