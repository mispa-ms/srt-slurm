#!/usr/bin/env bash
# Revert vllm#52388 (upstream PR #53774), for the 2026-08-28 nightly.
#
# #52388 replaced the per-group K3 Mamba aligned-state-index preparation with a
# fused multi-group kernel that caches raw block-table pointers. Those pointers
# are initialised during the temporary CUDA-graph memory-profile capture and
# stay cached after that allocation is released, so the later real capture
# dereferences stale addresses. Upstream reproduced the resulting illegal
# address on three separate H200 hosts (#85529 twice, #85541) and opened
# #53774 to revert while a safe pointer-lifetime design is worked out.
#
# On our GB300 disagg PP2 arm it surfaces first as a group-count disagreement
# rather than a bad address:
#
#   RuntimeError: Worker failed with error 'expected 3 block tables, got 4'
#     vllm/v1/worker/mamba_utils.py, from compile_or_warm_up_model
#
# The 08-19 images are unaffected because model_states/mamba_hybrid.py does not
# exist there -- the whole code path arrived with a9a17e7095 on 08-25.
#
# Drop this script once #53774 (or an equivalent fix) is in the nightly; the
# guard below will say so by refusing a forward apply that is already present.
set -euo pipefail

readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-revert-52388-per-group-k3-mamba-on-6f7df92a8.patch"
readonly MARKER="${VLLM_ROOT}/.k3_revert52388_applied"

if [[ -f "${MARKER}" ]]; then
  echo "[revert52388] already applied."
  exit 0
fi
if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[revert52388] FATAL: missing ${PATCH_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[revert52388] already reverted upstream in this image; nothing to do."
else
  echo "[revert52388] FATAL: the revert does not apply to this vLLM." >&2
  echo "[revert52388] Refusing: without it the second CUDA-graph capture reads" >&2
  echo "[revert52388] block-table pointers freed after the profile capture." >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi

touch "${MARKER}"
echo "[revert52388] applied."
