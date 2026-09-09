#!/usr/bin/env bash
# vllm#55745 (eastwood-c, opened 2026-09-07): record_stream idx_mapping in the PP
# draft broadcast. Fixes 09/08 "wall 3": broadcast_drafts gathers
# draft_tokens[input_batch.idx_mapping] on the side stream while idx_mapping is
# a per-step main-stream allocation, so the gather can read a freed/reused
# buffer -- a PP1-only ATen `vectorized_gather_kernel` device assert that
# vanishes under CUDA_LAUNCH_BLOCKING=1 (66840854 died, 66850730 with CLB
# completed). Reproduced on B200 AGG without any of our code. Carried verbatim.
# Same shape as vllm-container-deps-k3-ckptidx-829.sh.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-pr55745-pp-draft-broadcast-stream-on-9ea8f3ffc.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pr55745-908_applied"
if [[ -f "${MARKER}" ]]; then echo "[pr55745-908] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[pr55745-908] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pr55745-908] content already present in this image."
else
  echo "[pr55745-908] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[pr55745-908] applied."
