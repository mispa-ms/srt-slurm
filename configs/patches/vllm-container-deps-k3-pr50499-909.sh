#!/usr/bin/env bash
# vllm#50494 + #50499, refreshed head 0f6c6aa68f (2026-09-09 01:11 UTC, wire v13, packed MLA,
# per-region transfer geometry on top of merged #53780). vllm/ only, cut as
# `git diff c268198715 0f6c6aa68f -- vllm/`. Applies clean to nightly 385dce36 (2026-09-09).
# The 09-08 patch (head ce830b04) no longer applies: #53780 took NIXL_CONNECTOR_VERSION 11.
# Same shape as vllm-container-deps-k3-ckptidx-829.sh: forward dry-run, apply,
# or accept a reverse-applying (already present) tree; refuse anything else.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-pr50499-nixl-pp-hma-packed-on-385dce36-0f6c6aa.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pr50499-909_applied"
if [[ -f "${MARKER}" ]]; then echo "[pr50499-909] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[pr50499-909] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pr50499-909] content already present in this image."
else
  echo "[pr50499-909] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[pr50499-909] applied."
