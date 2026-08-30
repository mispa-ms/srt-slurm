#!/usr/bin/env bash
# The 08/30 chain plus a bound on the blocks #51358 pins.
# =============================================================================
# The 830 chain gets us onto the newest nightly. This adds the one experiment that
# says whether #51358's pin is why that nightly is 12.8% slower: 8,201.8 tok/s/chip
# against 9,409.6 on 08/28, with gpu_cache_hit_rate 71.6% against 86.2% and no change
# in pool size. See vllm-container-deps-k3-pincap.sh for the mechanism.
#
# Ordering: pincap edits the Mooncake store scheduler, which nothing else in the chain
# touches, and it is a no-op on any image without the pin -- so this chain is safe to
# point at 08/28 or 08/29 as well.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-830.sh
bash /configs/patches/vllm-container-deps-k3-pincap.sh
