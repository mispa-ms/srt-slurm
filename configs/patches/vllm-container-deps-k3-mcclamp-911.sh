#!/usr/bin/env bash
# vllm#51820 (open since 2026-08-11): clamp the Mooncake store's token_len to
# the hash coverage it actually has. Without it `store/data.py` asserts
# `token_len % hash_block_size == 0` and
# `token_len // hash_block_size <= len(block_hashes)`, which MTP and async
# scheduling can both violate -- the SA GB300 reference sweep runs mtp with a
# mooncake dram tier, so this is on the path for any arm that reproduces it.
# Wei Zhao named it as one of the two PRs GB300 still needs to run stock
# upstream nightlies.
#
# Only the vllm/ hunks are carried; the PR's test file is not in site-packages.
# Independent of every other step: it touches the mooncake store, nothing in
# the NIXL connector.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-pr51820-mooncake-token-len-clamp.patch"
readonly MARKER="${VLLM_ROOT}/.k3_mcclamp-911_applied"
if [[ -f "${MARKER}" ]]; then echo "[mcclamp-911] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[mcclamp-911] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[mcclamp-911] content already present in this image."
else
  echo "[mcclamp-911] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[mcclamp-911] applied."
