#!/usr/bin/env bash
# k3-merged-v4 + Mooncake under DCP, with the gather microbenchmark attached.
#
# Identical to kimi-k3-merged-v4-mooncake.sh in everything that touches the
# server. The only addition is the kernel timing run, which happens before the
# model loads and exists so the #51739 A/B has a direct measurement and not only
# an end-to-end one.
#
# The twin on the other side is kimi-k3-merged-v4-rev51739-mooncake.sh, which
# runs the same benchmark against the pre-#51739 kernel.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-merged-v4-mooncake.sh"
bash "$(dirname "${BASH_SOURCE[0]}")/k3_gather_microbench.sh"
