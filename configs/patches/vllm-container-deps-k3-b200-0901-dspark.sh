#!/usr/bin/env bash
# Kimi-K3 B200 on the 2026-09-01 nightly (7c5dc571).
# =============================================================================
# The 08/28 chain with three carries dropped and one added, all four verified
# against the 7c5dc571 tree before this was written.
#
# DROPPED, because upstream absorbed them -- each script detects this itself and
# says "already present", so they are still called and simply no-op:
#   #53324  Mooncake DCP + hybrid. connector.py now reads
#           `len(kv_cache_config.transfer_groups) > 1 and pcp > 1`, the post-fix
#           form with dcp gone from the condition.
#   #54167  low-latency GEMM init. _KimiK3LowLatencyApply.__init__ chains to
#           super().__init__().
#   #54044  mamba align metadata reset on profiling teardown (b383e16396).
#
# STILL OURS:
#   int64idx        no upstream PR. kda.py:281 still does state_idx *
#                   state_stride_0 with a 32-bit index, and the run dies around
#                   19.5 min without the cast.
#   dspark-pp-0901  vllm#50514 is still open, so spec-decode-under-PP stays a
#                   carry. Re-targeted, not re-derived: one hunk was dropped
#                   because upstream 6bafc049aa fixed the same bug by removing
#                   the term the hunk rewrote.
#
# ADDED:
#   tokenspeed-mtp-interleave   NEW WALL between 08/28 and 09/01. 7f4793eaa3
#                   ([Nixl][PD] DCP support for MLA models, #50611) promotes
#                   cp_kv_cache_interleave_size to block_size whenever a
#                   kv_connector is set -- which is true of our AGG arms, because
#                   MooncakeStoreConnector is a connector even without P/D. That
#                   trips the assert in cp_utils.py:41 for every speculative arm,
#                   because TokenSpeedMLAImpl does not declare the capability.
#                   Without this the ns arms die before serving.
# =============================================================================
set -euo pipefail

echo "=== k3-b200-0901: chain for the 2026-09-01 nightly ==="

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-pr54167.sh
bash /configs/patches/vllm-container-deps-k3-int64idx.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pp-0901.sh
bash /configs/patches/vllm-container-deps-k3-tokenspeed-mtp-interleave.sh
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh

# mambagroups2 is deliberately NOT in this chain. It patches the mamba align
# context so block tables are sliced with the context's own group ids, and it
# only matters for `mamba-cache-mode align` + DCP8 -- which these arms do not
# set, so it was inert here even on 08/28. On 7c5dc571 it is worse than inert:
# upstream restructured the align context, its anchor
# (initialize_from_forward_context) is gone, and its FATAL gate takes the whole
# run down before serving. Re-derive it only if an arm actually needs align mode.

echo "=== k3-b200-0901: done ==="
