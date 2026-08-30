#!/usr/bin/env bash
# v8, plus the two fixes a second pipeline stage needs.
# =============================================================================
# WHY THIS EXISTS. The B300 AGG ladder runs TP8 on eight chips because the model
# fits: K3 MXFP4 is 1,454 GiB, /8 = 181.7 GiB per chip against B300's 288. On
# B200 the same arithmetic lands above 178.35, which is why every B200 arm is
# TP8 x PP2 on sixteen chips -- there, PP2 is not a knob, it is the only layout
# that loads. This script exists so B300 can run that layout as a CHOICE, and
# find out whether the extra node buys anything above c70, where the eight-chip
# ladder turns over (12,074 -> 10,911 -> 7,952 at c70/c78/c86).
#
# TWO FIXES, BOTH LOAD-BEARING, NEITHER UPSTREAM. They come from the B200 PP2
# line, which has run them for days; they are carried rather than re-derived.
#
# 1. int64idx. _store_cache_checkpoints_kernel loads state_idx from an int32
#    block table and multiplies it by conv_state.stride(0) = 442,368, so the
#    product passes 2^31 at state_idx >= 4,855 and the store wraps negative.
#    PP stage 0 runs against higher block ids than stage 1 and is the one that
#    crosses: measured watermarks 4,859 (faults) against 4,835 (clean). It kills
#    the run with a CUDA illegal memory access about 19.5 minutes in -- long
#    after health passes, which is what makes it expensive to rediscover.
#    This is PP-triggered, not PP-caused, but PP is how we reach the threshold.
#
# 2. mambacache. _get_mamba_group_info caches the mamba group ids on self with
#    no key, and upstream #52388 added a second call site that populates it
#    during capture, earlier than the reader that used to. Each PP stage has its
#    own group layout -- 93 layers split two ways, 24 of them full attention --
#    so with PP2 the config the cache answers for is provably not the config in
#    hand, and capture dies on `expected 3 block tables, got 4`. This is the one
#    that is PP-caused rather than PP-triggered.
#
# NOT CARRIED, AND WHY. The B200 chain also runs dspark-pp-828 (which lifts an
# upstream refusal of PP combined with speculative decoding) and emptycache
# (which returns the draft model's cache before the symmetric-memory workspace
# is built). Both exist to make PP and DSpark coexist. Every B300 arm at these
# concurrencies is no-spec -- the frontier above c32 is `mcv8-dcp8-nospec-ts`,
# and spec has no arms above c48 on this workload because the draft KV halves
# the pool -- so neither has anything to do here, and carrying an unused patch
# would only widen what this arm is measuring.
#
# The Mooncake PP handshake fix is likewise not carried: it unblocks PP under
# DISAGGREGATED serving, where a pp_rank > 0 consumer must publish transfer
# metadata. The B200 AGG PP2 chain does not include it and runs Mooncake fine.
# =============================================================================
set -euo pipefail

HERE="$(dirname "${BASH_SOURCE[0]}")"

# Everything the eight-chip control runs: deps, the six commits, the marker
# assertions and the Mooncake DCP hit-boundary tests. Held identical on purpose
# -- the PP2 arms are meant to differ from their controls in the layout and in
# these two fixes, and in nothing else.
bash "$HERE/kimi-k3-nightly-v8.sh"

bash /configs/patches/vllm-container-deps-k3-int64idx.sh
bash /configs/patches/vllm-container-deps-k3-mambacache.sh

# Both scripts verify their own edit. What neither can see is whether this image
# refuses the layout outright, before any of it matters. PCP is refused for this
# model by a hard assert in the model file, and that refusal was found only after
# a submission; check the same shape for PP rather than assume it differs.
python3 - <<'PY'
import importlib.util
import pathlib
import re

root = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
k3 = root / "models/kimi_k3/nvidia"

refusals = []
for path in sorted(k3.glob("*.py")):
    for ln in path.read_text().splitlines():
        if re.search(r"pipeline_parallel_size\s*==\s*1", ln):
            refusals.append(f"{path.name}: {ln.strip()}")
assert not refusals, (
    "this image refuses pipeline parallelism for Kimi-K3 and the run would die "
    f"at engine init: {refusals}"
)

# The PCP assert IS expected to be here. If it ever disappears, that is worth
# knowing -- it is the one parallelism axis this model has never been able to
# use -- but it must not be read as licence to set PCP.
pcp = [ln.strip() for p in k3.glob("*.py")
       for ln in p.read_text().splitlines()
       if "prefill_context_parallel_size == 1" in ln]
print(f"=== pp2 preflight: no PP refusal; PCP still refused ({len(pcp)} sites) ===")
PY

echo "=== v8-pp2 ready: v8 + int64idx + mambacache ==="
