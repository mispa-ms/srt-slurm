#!/usr/bin/env bash
# Ours: MooncakeStoreConnector no-op set_xfer_handshake_metadata_pp_aware. Without it the base guard raises 'received pp_rank > 0 handshake metadata' and PP-disagg with a mooncake tier dies at engine-core init.
# Same shape as vllm-container-deps-k3-ckptidx-829.sh: forward dry-run, apply,
# or accept a reverse-applying (already present) tree; refuse anything else.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-mooncake-store-pp-handshake-override.patch"
readonly MARKER="${VLLM_ROOT}/.k3_mcpp-908_applied"
if [[ -f "${MARKER}" ]]; then echo "[mcpp-908] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[mcpp-908] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[mcpp-908] content already present in this image."
else
  echo "[mcpp-908] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[mcpp-908] applied."
