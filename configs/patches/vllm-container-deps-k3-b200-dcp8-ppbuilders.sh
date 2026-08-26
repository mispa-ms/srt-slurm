#!/usr/bin/env bash
# The B200 DCP8 chain plus one metadata builder per pipeline stage.
# =============================================================================
# Tests whether the illegal memory access on PP2 is the shared attention-metadata
# builder. See vllm-container-deps-k3-ppbuilders.sh for the full reasoning; in short,
# the V2 runner hardcodes one builder and index 0, the builder owns the chunked-prefill
# workspace and MLA's prepared metadata, and PP keeps pp_size microbatches in flight.
#
# Same image, same DCP8, same three direct flags, same Mooncake as its parent
# -sasrt-latest arm, which faulted after 230 of 838 requests. The only difference is
# that consecutive builds no longer share buffers.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-ppbuilders.sh
