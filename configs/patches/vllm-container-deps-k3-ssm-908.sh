#!/usr/bin/env bash
# Ours: route Mamba/SSM members through #50499's member-identity path. #50499 gates it off with 'and not self._has_mamba' and refuses PP with Mamba at init. Must follow pr50499-908.
# Same shape as vllm-container-deps-k3-ckptidx-829.sh: forward dry-run, apply,
# or accept a reverse-applying (already present) tree; refuse anything else.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-ssm-members-on-9ea8f3ffc-pr50499.patch"
readonly MARKER="${VLLM_ROOT}/.k3_ssm-908_applied"
if [[ -f "${MARKER}" ]]; then echo "[ssm-908] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[ssm-908] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[ssm-908] content already present in this image."
else
  echo "[ssm-908] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[ssm-908] applied."
