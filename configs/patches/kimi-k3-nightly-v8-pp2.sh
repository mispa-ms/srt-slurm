#!/usr/bin/env bash
# v8, plus the three fixes a second pipeline stage needs.
# =============================================================================
# WHY THIS EXISTS. The B300 AGG ladder runs TP8 on eight chips because the model
# fits: K3 MXFP4 is 1,454 GiB, /8 = 181.7 GiB per chip against B300's 288. On
# B200 the same arithmetic lands above 178.35, which is why every B200 arm is
# TP8 x PP2 on sixteen chips -- there, PP2 is not a knob, it is the only layout
# that loads. This script exists so B300 can run that layout as a CHOICE, and
# find out whether the extra node buys anything above c70, where the eight-chip
# ladder turns over (12,074 -> 10,911 -> 7,952 at c70/c78/c86).
#
# THREE FIXES, ALL LOAD-BEARING, NONE UPSTREAM. They come from the B200 PP2
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
# 3. dspark-pp-828. The nightly refuses speculation under PP outright --
#    `{method} with pipeline parallel is not supported` in model_runner.py --
#    and nobody upstream runs K3 with PP, so nobody upstream is lifting it. It
#    narrows the refusal to models that cannot forward aux hidden states across
#    stages, keeps a hard refusal above pp=2, carries the target's aux taps
#    alongside IntermediateTensors, and loads the real embedding table on the
#    last stage where the target's embed is a PPMissingLayer.
#
# ONE CHAIN, INCLUDING FOR THE NO-SPEC ARMS, AND THAT IS DELIBERATE. Half the
# arms on this script never configure speculation, and the reflex is to keep an
# unused patch off them. Read against this image, dspark-pp-828 executes nothing
# without a speculative config:
#
#   compute_need_sampled_mask  the new branch is gated on
#                              `num_draft_tokens_per_req is not None`
#   PPSlot.draft_tokens        defaults None; the extra output is added only
#                              when it is not
#   pack/unpack/make_empty_aux_hidden_states
#                              all iterate aux_hidden_state_layers, which is ()
#                              unless EAGLE-style drafting set it -- zero tensors
#                              added, so the PP transport buffers are unchanged
#   speculative.py, dspark/utils.py, eagle3_utils.py
#                              not reached
#
# So there is nothing to price, and splitting the chain would cost something
# real: the no-spec points are the control for the speculative ones, and a
# control that runs a different build is a weaker control. Same build for all
# six; each is read as a delta against its own eight-chip PP1 point.
#
# STILL NOT CARRIED. emptycache, which returns the loader's cached blocks to the
# driver before the draft's MLA builds a second direct-DCP symmetric-memory
# workspace. Its own header says B300 does not hit that: 131.57 GiB of weights
# against 288 rather than 180, so the loader's cache never reaches the ceiling,
# and under PP2 each chip holds half the layers. It would also edit
# model_runner.py, which dspark-pp-828 rewrites.
#
# Nor the Mooncake PP handshake fix: it unblocks PP under DISAGGREGATED serving,
# where a pp_rank > 0 consumer must publish transfer metadata. The B200 AGG PP2
# chain does not include it and runs Mooncake fine.
# =============================================================================
set -euo pipefail

HERE="$(dirname "${BASH_SOURCE[0]}")"

# Everything the eight-chip control runs: deps, the six commits, the marker
# assertions and the Mooncake DCP hit-boundary tests. Held identical on purpose
# -- the PP2 arms are meant to differ from their controls in the layout and in
# these three fixes, and in nothing else.
bash "$HERE/kimi-k3-nightly-v8.sh"

bash /configs/patches/vllm-container-deps-k3-int64idx.sh
bash /configs/patches/vllm-container-deps-k3-mambacache.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pp-828.sh


# The three scripts verify their own edits. This asks the different question:
# does the image refuse the layout outright, before any of it matters. Kept in
# its own file because the first inline version matched a substring rather than
# a statement and failed six healthy jobs -- the file's docstring is that story.
python3 /configs/patches/k3-pp2-preflight.py

echo "=== v8-pp2 ready: v8 + int64idx + mambacache + dspark-pp-828 ==="
