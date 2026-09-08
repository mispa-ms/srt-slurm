#!/usr/bin/env bash
# vllm#50494 + #50499 (head ce830b04, 2026-09-05): NIXL member-identity routing for PP push. Both still open; applies clean to nightly 9ea8f3ffc3.
# Same shape as vllm-container-deps-k3-ckptidx-829.sh: forward dry-run, apply,
# or accept a reverse-applying (already present) tree; refuse anything else.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-pr50499-nixl-pp-hma-packed-on-9ea8f3ffc.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pr50499-908_applied"
if [[ -f "${MARKER}" ]]; then echo "[pr50499-908] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[pr50499-908] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pr50499-908] content already present in this image."
else
  echo "[pr50499-908] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[pr50499-908] applied."
