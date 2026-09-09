#!/usr/bin/env bash
# Ours: route Mamba/SSM members through #50499's member-identity path, re-derived on the
# refreshed #50499 head 0f6c6aa68f (per-member block counts). #50499 gates it off with
# 'and not self._has_mamba' and refuses PP with Mamba at init. Must follow pr50499-909.
# vllm branch misunp/k3-ssm-members-on-50499-0f6c6aa (0c44504421), vllm/ only.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-ssm-members-on-385dce36-pr50499-0f6c6aa.patch"
readonly MARKER="${VLLM_ROOT}/.k3_ssm-909_applied"
if [[ -f "${MARKER}" ]]; then echo "[ssm-909] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[ssm-909] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[ssm-909] content already present in this image."
else
  echo "[ssm-909] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[ssm-909] applied."
