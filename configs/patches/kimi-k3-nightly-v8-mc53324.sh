#!/usr/bin/env bash
# ARM 1: our Mooncake DCP carry replaced by upstream vllm#53324, nothing else.
# =============================================================================
# The one thing that moves against the v8 control is which implementation of the
# Mooncake DCP hit-boundary fix is in the image. Both fix the same defect. Ours
# revalidates the reconciled boundary and steps the hit back until its key
# exists; #53324 keeps the longer hit and resolves, per group, the hash boundary
# whose key actually stored that tail block. Ours throws away a hit upstream can
# now load -- a claim written in test_mooncake_dcp_keyset.py's header when we
# wrote the carry and never measured until this arm.
#
# It is also the prerequisite for Wei's compact group I/O, which is built on
# #53324 and cannot sit on ours. Priced separately so that if the pair moves,
# the reason is not ambiguous.
#
# #53324 merged 2026-08-29 02:10 UTC, one day after this stack's 6f7df92a8e
# nightly. Applied as the upstream commit verbatim.
set -euo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"

# The carry without its three Mooncake files, and the Mooncake assertions moved
# to the script below, which knows which contract is live.
export K3_OURS_PATCH=/configs/patches/k3-ours6-v8-nomc.patch
export K3_SKIP_MOONCAKE_CHECKS=1

bash "$HERE/kimi-k3-nightly-v8.sh"
bash /configs/patches/vllm-container-deps-k3-upstream-53324.sh
echo "=== v8-mc53324 ready: v8 carry (no mooncake) + upstream #53324 ==="
