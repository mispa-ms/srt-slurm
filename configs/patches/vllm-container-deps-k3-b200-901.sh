#!/usr/bin/env bash
# The 2026-09-01 nightly (7c5dc571), with the full DSpark + PP + Mooncake stack.
# =============================================================================
# WHY MOVE OFF 08/28. Four merged commits since our pinned image land on items this
# arm actually spends time in, measured from its own nsys trace rather than assumed:
#
#   #54277  FlashInfer MLA for DSpark drafting under DCP   (08-29)
#   #54261  native CUDA AttnRes as the SM100 default        (08-31)
#            -> attn_res_fwd_online_v2_kernel, 1.319 ms/step over 88 launches
#   #53382  cooperative topk tuned for medium batch sizes   (08-31)
#            -> routingIndicesBlockScoresKernel, 0.635 ms/step over 44 launches
#   #54040  retire the DSv3 router GEMM kernel              (08-31)
#
# #54277 matters twice. It is the commit that adds
# `supports_non_causal_multi_token_dcp = True` to flashinfer_mla.py -- the exact gate
# that rejected our FLASHINFER_MLA arm on 08/28. On this image the MLA decode backend
# becomes an A/B axis again, and MLA decode is the largest single item on our critical
# path at 15.2% of stream-23 kernel time.
#
# WHAT IT COSTS. #51358 is also in this image, and it measured -9.0% applied to 08/28
# and -12.8% as shipped in the 08/30 nightly. That is not a regression to be reverted:
# the pre-patch path resolved positionally from an append-only mirror of an in-place
# mutated table, so the throughput it bought was partly unsound reads. So this arm is a
# NET measurement -- new kernels against that known loss -- and the 08/30 number is not
# the answer, because 08/30 had neither the four commits above nor the 48,45 split.
#
# WHAT CHANGED IN THE STACK. k3-dspark-pp is rebased: upstream deleted the
# `not_finishing` early-finish condition that one of its hunks patched, so that hunk is
# dropped and the other 23 apply at fuzz 0. int64idx and mambacache are both still
# needed -- checked against 7c5dc571, the int32 state_idx load and the unkeyed
# _get_mamba_group_info are unchanged. pr54167 is already upstream here and its script
# detects that and skips. The interleave narrowing is required: #50611's promotion is
# in this image and its anchor still matches exactly once.
# =============================================================================
set -euo pipefail

echo "=== k3-b200-901: 09/01 nightly stack ==="
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-pr54167.sh
bash /configs/patches/vllm-container-deps-k3-int64idx.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pp-901.sh
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh
bash /configs/patches/vllm-container-deps-k3-mambacache.sh
bash /configs/patches/vllm-container-deps-k3-interleave.sh
echo "=== k3-b200-901: done ==="
