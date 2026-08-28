#!/usr/bin/env bash
# The B200 DCP8 chain, upstream #53614, and the repair its own assert needs.
# =============================================================================
# #53614 applied cleanly and then died on its own assert five minutes into serving,
# before the IMA window opened, so that arm measured nothing. This chain adds the
# minimal repair -- the two asserts become the `return None` the same function already
# uses twenty lines below -- plus counters, because a guard that always fires would make
# the patch a no-op and a clean run meaningless.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-pr53614.sh
bash /configs/patches/vllm-container-deps-k3-pr53614-nullguard.sh
