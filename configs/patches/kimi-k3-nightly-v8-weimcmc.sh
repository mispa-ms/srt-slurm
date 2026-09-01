#!/usr/bin/env bash
# ARM 3: the disagg stack -- everything in -weimc, plus MultiConnector load
# ownership, which only the disagg recipe can reach.
# =============================================================================
# Read kimi-k3-nightly-v8-weimc.sh first. This adds one thing to it, and the
# thing is only reachable when MooncakeStoreConnector is a child of
# MultiConnector -- which is the disagg recipe (Nixl for the P-to-D handshake,
# Mooncake for external DRAM KV) and never the aggregated one.
#
# It is also the one re-derived patch on this stack: Wei's bb9afd157 is written
# against the pre-#53324 form of the line it changes. Its own file says exactly
# what was decided and why.
#
# WHY DISAGG IS WHERE THIS BELONGS. Wei's measurements are on a disagg run --
# DCP8 + DEP16 + DSpark4, Nixl push plus Mooncake -- and the effect he measured
# depends on a page width our aggregated arms do not have. DSpark's five draft
# MLA layers widen the shared physical page from 24 slots to 29, and a KDA group
# has 23 real layers, so the cross-group padding a compact write removes is six
# slots of 29 (20.7%) with DSpark and one of 24 (4.2%) without. Our aggregated
# arms are nospec: they carry the same code and about a fifth of the effect.
# Confirmed in our own logs -- `num_groups=4, num_segments=24` on a nospec run,
# `num_segments=29` on a DSpark one.
#
# And the pressure is real here: our 1P1D producer's Mooncake master reaches
# memory_ratio 0.941 against a 0.95 high watermark and evicts on every arm from
# c32 up. Compact I/O has something to win only where the store is full, and it
# is full.
set -euo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"

bash "$HERE/kimi-k3-nightly-v8-weimc.sh"
bash /configs/patches/vllm-container-deps-k3-wei-multiconnector-load.sh
echo "=== v8-weimcmc ready: v8 (no mooncake) + #53324 + compact I/O + MC load ==="
