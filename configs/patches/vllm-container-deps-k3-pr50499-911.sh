#!/usr/bin/env bash
# vllm#50494 + #50499, head d1ac007938 (2026-09-10, rebased onto the renamed
# #50494 6209b817c9: _member_* -> _transfer_layer_*, region_members -> region_layers).
# Cut as the vllm/ delta of that stack cherry-picked onto nightly e7edf17ce.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-pr50499-nixl-pp-hma-packed-on-e7edf17ce-d1ac007.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pr50499-911_applied"
if [[ -f "${MARKER}" ]]; then echo "[pr50499-911] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[pr50499-911] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pr50499-911] content already present in this image."
else
  echo "[pr50499-911] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[pr50499-911] applied."
