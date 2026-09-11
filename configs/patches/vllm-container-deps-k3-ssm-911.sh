#!/usr/bin/env bash
# Ours: route Mamba/SSM layers through #50499's layer-name path, re-derived on the
# rebased head d1ac007938 (new vocabulary). #50499 gates it off with
# 'and not self._has_mamba' and refuses PP with Mamba at init.
# vllm branch misunp/k3-ssm-layers-on-50499-d1ac007 (bb7e53dc57). Must follow pushdcp-911.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-ssm-layers-on-e7edf17ce-pr50499-d1ac007.patch"
readonly MARKER="${VLLM_ROOT}/.k3_ssm-911_applied"
if [[ -f "${MARKER}" ]]; then echo "[ssm-911] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[ssm-911] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[ssm-911] content already present in this image."
else
  echo "[ssm-911] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[ssm-911] applied."
