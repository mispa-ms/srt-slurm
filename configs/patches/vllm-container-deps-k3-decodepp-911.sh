#!/usr/bin/env bash
# Ours: pipeline-parallel decode on the push path. Upstream refuses a consumer with
# pipeline_parallel_size > 1 (completions counted per consumer rank, and the transfer
# handles were keyed by TP rank alone so every decode stage collapsed onto stage 0).
# Keys dst_xfer_side_handles by (remote_pp_rank, remote_tp_rank), carries decode_pp_size
# on PUSH_REG, and fans WRITEs out over the overlapping decode stages. Also the reverse
# alignment: an unsharded (PP1) producer builds per-stage descriptors and a per-stage
# local handle for each stage of a PP-sharded consumer (P PP1 -> D PP2 could not
# handshake before: the stage advertises only its own layer window).
# A PP-sharded producer handshakes only with the decode stages it writes to (P PP2 ->
# D PP2 handshook with both and the alignment refused the other half, 67584704).
# pp_rank is set in __init__ (the decode side's PUSH_REG handshake raised AttributeError
# on it in 67594496, so no request ever reached the producer).
# vllm branch misunp/k3-decode-pp-on-50499-d1ac007 (see git log). Must follow ssm-911.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-decode-pp-on-e7edf17ce.patch"
readonly MARKER="${VLLM_ROOT}/.k3_decodepp-911_applied"
if [[ -f "${MARKER}" ]]; then echo "[decodepp-911] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[decodepp-911] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[decodepp-911] content already present in this image."
else
  echo "[decodepp-911] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[decodepp-911] applied."
