#!/usr/bin/env bash
# Round 3 of the vllm#51766 cost hunt: stop counting, take a trace. #51766
# reverted here.
#
# WHY A TRACE AND NOT MORE COUNTERS. Rounds 1 and 2 each tested a mechanism and
# each refuted it. Round 1: the same-step CoW deferral fires 9 times in 39,500
# scheduler steps, and skipping instead of breaking the waiting loop moved
# throughput +1.5% / -0.3%. Round 2: every event on both CoW branches counted
# with the fix present and reverted, and per step all six counters match -- the
# branch split is 56% against 55% and tail_offloads tracks cow_running exactly,
# so there is no extra Mooncake traffic either.
#
# What round 2 did establish is the shape of the problem. The reverted engine
# completed 11.6% more scheduler steps in the same hour at c70 with identical
# per-step event counts, so the cost is not extra events -- EACH STEP IS ABOUT
# 10% SLOWER. Hand-placed timers would still assume we know which step to time,
# and reading the source says the opposite: with num_speculative_blocks = 0 both
# consumers of _allocated_block_reqs compute the same value, so on a no-spec arm
# the one line should change nothing at all. It reproducibly costs 8.8%.
#
# So the model of the code is wrong somewhere, and the way to find out where is
# to look at what the GPU and the host actually did rather than at what the
# source says they should have.
#
# THE TWO TRACES MUST BE READ AS A PAIR. Same concurrency, same image, same
# window placement, one line apart. Either alone says nothing.
#
# WINDOW PLACEMENT, measured off the round-2 runs rather than guessed. On
# c70-mcv9f the worker log starts at 17:36:45, AgentX warmup at 17:52:59 and the
# reported window runs 18:19:19 to 19:19:19. Mid-window is 4,354 s after the
# worker starts; the config asks for 4,400 s and captures 30 s. The window is an
# hour wide, so ten minutes of startup variance either way still lands inside.
#
# -t cuda-sw,nvtx is set in the config, not here: on Blackwell the default
# "cuda" uses hardware-event tracing, which kills the worker async output copy
# stream at capture teardown.
set -euo pipefail

# vllm/vllm-openai ships no nsys, so the wrapped worker launch would exit 127.
bash "$(dirname "${BASH_SOURCE[0]}")/install-nsys-cli.sh"

# Everything else -- deps, our eight, the counters, and #51766 reverted -- comes
# from the round-2 script, including its assertion that this arm is the side it
# claims to be.
bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-nightly-v9-rev.sh"

command -v nsys >/dev/null || {
    echo "ERROR: nsys is not on PATH after install-nsys-cli.sh" >&2
    exit 1
}
nsys --version
