#!/usr/bin/env bash
# v8-pp2, plus the patch that lets speculation and a pipeline stage coexist.
# =============================================================================
# WHY BOTH AXES AT ONCE, WHICH IS NORMALLY THE WRONG THING TO DO. The B300 AGG
# speculative arm does not merely fall off above c32, it collapses:
#
#   mcv8-dcp8-dspark7   c32 8,312   c40 6,910   c48 5,195   c56 4,371   c64 3,698
#   mcv8-dcp8-nospec-ts c32 10,159  c40 10,477  c48 11,372  c56 11,790  c64 11,601
#
# At c48 speculation costs 54%. The standing explanation is that the draft model's
# KV halves a pool that is already the binding constraint at that concurrency, and
# every arm above c32 has been no-spec since. If that explanation is right, then
# PP2 -- which halves the weights each chip carries and roughly doubles the pool
# per chip -- is aimed at exactly the thing that broke, and the speculative arm is
# where it should show up first. The no-spec PP2 points are the control for that:
# there the pool was never binding, so PP2 should buy little or nothing.
#
# So this is not two knobs turned hoping one lands. It is one hypothesis with a
# prediction on each side, and the no-spec arms falsify it if they gain as much.
#
# WHAT THIS ADDS OVER v8-pp2. One patch, dspark-pp-828, 7 files and 596 lines,
# keyed to this image at fuzz 0. The nightly refuses the combination outright --
# `{method} with pipeline parallel is not supported` in model_runner.py -- and
# nobody upstream runs K3 with PP, so nobody upstream is lifting it. The patch
# replaces the blanket refusal with one that fires only when the model cannot
# forward aux hidden states across stages, keeps a hard refusal above pp=2
# (a middle stage has never run on hardware and its failure mode is degraded
# acceptance rather than a crash, which is the worst kind here), carries the
# target's aux taps alongside IntermediateTensors, and loads the real embedding
# table on the last stage where the target's embed is a PPMissingLayer.
#
# WHAT IS STILL NOT CARRIED, AND WHY IT IS NOT AN OVERSIGHT. The B200 chain pairs
# this patch with `emptycache`, which returns the loader's cached blocks to the
# driver before the draft model builds its own direct-DCP symmetric-memory
# workspace. That script says in its own header why B300 does not need it: the
# same 131.57 GiB of weights sit against 288 GiB rather than 180, so the loader's
# cache never reaches the ceiling and the driver still has room when the draft
# asks. Under PP2 each chip carries half the layers, so the margin is larger
# still. Carrying it would add an edit to model_runner.py -- the file this
# patch rewrites -- for a failure that cannot occur here.
#
# ACCEPTANCE IS AN INPUT ON THESE ARMS, NOT A MEASUREMENT. rejection_sample_method
# is synthetic, so the run reports the acceptance length it was handed (3.84 for
# k=7) whether or not the draft is working. That is what makes this a throughput
# experiment and not an accuracy one, and it is why the PP+spec patch's own note
# about a middle stage failing as "degraded acceptance" matters: on a synthetic
# arm that failure is invisible. pp=2 has no middle stage, and the patch refuses
# anything larger, which is the only reason this is safe to read.
# =============================================================================
set -euo pipefail

HERE="$(dirname "${BASH_SOURCE[0]}")"

# v8 + int64idx + mambacache + the PP-refusal preflight.
bash "$HERE/kimi-k3-nightly-v8-pp2.sh"

bash /configs/patches/vllm-container-deps-k3-dspark-pp-828.sh

# The wrapper checks its own five call sites. This checks the property that made
# the patch necessary: the engine must now accept pp=2 with a draft model, and
# must still refuse pp=3, because the guard that stops a middle stage is the only
# thing standing between a synthetic-acceptance run and a silently wrong number.
python3 - <<'PY'
import importlib.util
import pathlib

root = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
mr = (root / "v1/worker/gpu/model_runner.py").read_text()

assert "pipeline_parallel_size=2" in mr, (
    "the pp>2 refusal is not in model_runner.py; without it a middle stage could "
    "run, and under synthetic acceptance its failure would not be visible"
)
assert "supports_aux_hidden_states_over_pp" in mr, (
    "the narrowed refusal is missing; the blanket one may have been removed "
    "without its replacement"
)
print("=== pp2-dspark preflight: pp=2 accepted, pp>2 still refused ===")
PY

echo "=== v8-pp2-dspark ready: v8 + int64idx + mambacache + dspark-pp-828 ==="
