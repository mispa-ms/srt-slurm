#!/usr/bin/env bash
# The chain the latest nightly actually needs for a DSpark + PP2 + DCP8 arm.
# =============================================================================
# Two of the patches this branch used to carry are gone, because a9a17e7095 has them:
#
#   mooncake #53324   the guard that refused DCP is fixed upstream
#   graphpool #53682  the PP2 startup deadlock is fixed upstream
#
# And #53614 is not here on purpose. It looked like the IMA fix and was ruled out by
# measurement -- active for 86.3% of checkpoint attempts, fault unchanged at 19.4 min
# against 19.5 without it -- and it separately dies on an assert it adds itself.
#
# What remains is all PP, which is the part nobody upstream runs:
#
#   k3-b200-dcp8        the DCP dummy-batch fix; without it every DP-idle step reads a
#                       dcp_local_seq_lens the dummy path never set
#   int64idx            state_idx to int64; without it the run dies at ~19.5 min
#   dspark-pp-nightly   lifts the PP+spec refusal, rebased onto this nightly
#   emptycache          returns the loader's cached blocks before the draft is built
#
# hfshim comes with k3-b200-dcp8. emptycache is chained last because it is the one
# thing 728d3ad does not carry either, so it is needed on both images.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-int64idx.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pp-nightly.sh
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh
