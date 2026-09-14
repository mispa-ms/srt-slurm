#!/usr/bin/env bash
# Cap the size of a single NIXL memory region (VLLM_NIXL_MAX_MR_BYTES).
#
# allocate_kv_cache puts every layer in one backing allocation -- the assert
# above it says so -- and the NIXL worker registers that allocation by
# storage.nbytes(), keyed by (storage_addr, mem_type). registration_ranges
# therefore collapses to ONE entry and a prefill stage here registers a single
# ~147 GiB memory region (decode ~47 GiB). Wei Zhao traced silent RDMA data
# corruption on this cluster to an oversized GPU dma-buf MR crossing a CUDA VMM
# boundary at GPU VA 0xff8000000000: the write reports success and the peer
# receives wrong bytes, and because the resulting NaNs skewed MoE routing the
# CORRUPTED arm looked 20% faster. Mooncake hit it at 56.95 GiB; ours is 2.6x
# that. His fix was 14 MRs of <= 4 GiB.
#
# This step only adds the knob. The default (0) registers whole allocations
# exactly as today, so an arm without VLLM_NIXL_MAX_MR_BYTES in its environment
# is byte-for-byte the old behaviour and the A/B is one variable.
#
# Cuts are taken on block boundaries only, so every transfer descriptor still
# resolves inside exactly one registered range.
#
# Independent of ssm, evict and dpp. The split reads block_stride_per_layer and
# region_num_blocks, which the unconditional layer-name steps (pr50499-911,
# pushdcp-911) already provide -- checked on a K3_OURS=mrcap tree. Cut against
# the ssm base so it composes with all three.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-mrcap-on-dc36fcce.patch"
readonly MARKER="${VLLM_ROOT}/.k3_mrcap-911_applied"
if [[ -f "${MARKER}" ]]; then echo "[mrcap-911] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[mrcap-911] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[mrcap-911] content already present in this image."
else
  echo "[mrcap-911] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[mrcap-911] applied."
