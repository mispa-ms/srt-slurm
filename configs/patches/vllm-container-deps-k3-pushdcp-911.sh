#!/usr/bin/env bash
# Ours: lift #50611's push+DCP refusal for the symmetric case and enforce equal
# decode_context_parallel_size at handshake. Smaller than the 09-08 version:
# vllm#55531 (merged 2026-09-10) already removed the worker hybrid+DCP refusal,
# so only the NixlPushConnector.__init__ raise and the handshake check remain.
# vllm branch misunp/nixl-push-symmetric-dcp-v2 (59e8eaf465). Must follow pr50499-911.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-nixl-push-symmetric-dcp-on-e7edf17ce.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pushdcp-911_applied"
if [[ -f "${MARKER}" ]]; then echo "[pushdcp-911] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[pushdcp-911] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pushdcp-911] content already present in this image."
else
  echo "[pushdcp-911] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[pushdcp-911] applied."
