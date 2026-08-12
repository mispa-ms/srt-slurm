#!/usr/bin/env bash
# Runtime setup for k3-merged-v5: our eight commits on 2026-08-12 upstream.
#
# v5 is v4's branch rebased from e3fe212eaf onto 8151f2ad43, 50 commits later.
# It is eight commits, not nine, because upstream landed its own version of the
# fix we were carrying as vllm#50532: 0a94d85a66 (#51865) requires all requests
# to be decoding before uniform-decode dispatch, which is the same defect with a
# `has_prefill` flag where ours passed the computed-token arrays. Ours was
# skipped rather than merged, the same way #50484's cherry-pick was dropped at
# v4 once it landed.
#
# WHY THIS DOES NOT CHAIN kimi-k3-merged-v2.sh LIKE ITS PREDECESSORS.
# That script asserts `get_uniform_decode_token_count` appears in
# vllm/v1/worker/gpu/cudagraph_utils.py. #51917 unified that helper and moved
# its use into model_runner.py, so the marker is now absent from a perfectly
# correct image. The chain encoded a history that has stopped being true, and
# an assertion that fails on a good image is worse than no assertion. The marker
# set below is the same idea rebuilt against what v5 actually contains.
#
# WHAT THE REBASE HAD TO RESOLVE, and why it is worth knowing before reading a
# number from this image. #51843 rewrote the same block as our #50493 carry:
# upstream keeps fine-grained prefix-cache hits gated on `dcp_world_size == 1`
# and adds a check that every group's manager can serve a fine-grained lookup;
# our carry removes the DCP gate because DCP scales the effective full-attention
# block instead. The resolution keeps both -- the DCP relaxation and upstream's
# manager-capability guard. That guard is a no-op for this model: both
# FullAttentionManager and MambaManager declare
# supports_fine_grained_hash_lookup, so the 24 MLA + 69 KDA layout is unaffected.
# If a future group type does not, this arm will quietly lose partial hits, and
# the warning it logs is the thing to grep for.
set -euo pipefail

export FI_VER=0.6.16.post3
bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-aggv2.sh"

python3 - <<'PY'
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent

# (path, marker, what it proves)
CHECKS = [
    # our six, the ones absent from upstream
    ("v1/worker/gpu/cp_utils.py", "def cp_local_slot(", "DSpark under DCP"),
    ("v1/attention/backends/mla/tokenspeed_mla.py", "VLLM_TS_MLA_DCP_FLATTEN",
     "TokenspeedMLA DCP flatten flag"),
    # upstream's, which separate v5 from every earlier image on this track
    ("v1/worker/mamba_utils.py", "_memcpy_u64_tiled", "#49436, absent from v2"),
    ("v1/worker/gpu/spec_decode/autoregressive/speculator.py",
     "_fused_multi_step_decode", "#46849, absent from v3"),
    ("v1/worker/utils.py", "def get_uniform_decode_token_count",
     "#51865, upstream's replacement for our #50532 carry"),
]
missing = [(rel, who) for rel, mark, who in CHECKS
           if not (root / rel).exists() or mark not in (root / rel).read_text()]
assert not missing, f"wrong image: expected k3-merged-v5, missing {missing}"

# #51865 is upstream's, not ours: it takes has_prefill rather than the arrays
# our carry passed. If a rebase ever reinstates ours on top, the signature is
# where that shows, and the two must not both be present.
import inspect  # noqa: E402

from vllm.v1.worker.utils import get_uniform_decode_token_count  # noqa: E402

params = list(inspect.signature(get_uniform_decode_token_count).parameters)
assert params[-1] == "has_prefill", (
    f"get_uniform_decode_token_count takes {params}, which is not upstream's "
    "#51865 form -- our skipped #50532 carry may have been reapplied"
)

# The #50493/#51843 resolution: the DCP gate is gone and the capability guard
# stays. Assert the guard is present rather than its outcome, which is
# config-dependent.
coord = (root / "v1/core/kv_cache_coordinator.py").read_text()
assert "supports_fine_grained_hash_lookup" in coord, (
    "vllm#51843's manager-capability guard is missing; the #50493 resolution "
    "dropped the DCP gate on the assumption that guard replaces it"
)
assert "dcp_world_size == 1 and has_partial_mamba_group" not in coord, (
    "the dcp_world_size == 1 gate is back, so vllm#50493's relaxation is not in "
    "effect and partial prefix-cache hits are off under DCP"
)

import flashinfer  # noqa: E402
print(f"=== v5 image verified; flashinfer {flashinfer.__version__} ===")
PY
