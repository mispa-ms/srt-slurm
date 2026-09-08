#!/usr/bin/env bash
# Ours: allow NixlPushConnector with DCP when both instances shard identically.
# vllm#50611 (merged 2026-08-29) added DCP to the pull path by reading each
# block's slice from several producer ranks, and refused push + DCP and
# hybrid + DCP outright. Push never slices -- rank i writes to rank i -- and
# for equal TP/DCP #50611's own dcp_source_ranks() reduces to the counterpart
# rank, so the refusals are conservative for the symmetric case. This lifts
# them for push only and enforces DCP equality at handshake. Our 08-19/08-29
# stacks ran exactly this shape (TP8xDCP8 -> TP8xDCP8, GSM8K 0.951).
# Same shape as vllm-container-deps-k3-ckptidx-829.sh.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-nixl-push-symmetric-dcp-on-9ea8f3ffc.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pushdcp-908_applied"
if [[ -f "${MARKER}" ]]; then echo "[pushdcp-908] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[pushdcp-908] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pushdcp-908] content already present in this image."
else
  echo "[pushdcp-908] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[pushdcp-908] applied."
