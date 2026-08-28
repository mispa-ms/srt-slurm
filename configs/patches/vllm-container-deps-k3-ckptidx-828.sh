#!/usr/bin/env bash
# prefill-checkpoint index int64 cast, for the 2026-08-28 nightly (6f7df92a).
# Idempotent: a marker file records the apply, and the patch is checked for
# reverse-applicability before anything is written.
set -euo pipefail

readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-prefill-checkpoint-index-int64-on-6f7df92a8.patch"
readonly MARKER="${VLLM_ROOT}/.k3_ckptidx-828_applied"

if [[ -f "${MARKER}" ]]; then
  echo "[ckptidx-828] already applied."
  exit 0
fi
if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[ckptidx-828] FATAL: missing ${PATCH_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[ckptidx-828] content already present in this image."
else
  echo "[ckptidx-828] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  echo "[ckptidx-828] a silently unpatched image fails later and less legibly." >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi

touch "${MARKER}"
echo "[ckptidx-828] applied."
