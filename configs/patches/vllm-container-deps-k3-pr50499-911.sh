#!/usr/bin/env bash
# vllm#50494 + #50499, head bc2e6269e4 (2026-09-14, rebased for the upstream
# registration_ranges reshape: its values went (addr, nbytes, dev, meta) ->
# (start, end, device_id)). The head is a direct DESCENDANT of nightly
# dc36fcce, so the patch is the stack's own commits as a diff and applies by
# construction; #56746 (grpc_server move) sits in between and is excluded by
# restricting the diff to the nixl connector directory.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-pr50499-nixl-pp-hma-packed-on-dc36fcce-bc2e626.patch"
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
