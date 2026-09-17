#!/usr/bin/env bash
# Ours: log why a lookup that FINDS blocks still loads none.
#
# find_longest_cache_hit counts a contiguous prefix from block 0, so a present
# set with an early hole returns 0 however many blocks it holds. This logs
# hit_length, how many of the request's blocks the store has, the index of the
# first one it does not, and a bitmap of the first 16 -- next to the pool the
# lookup just built.
#
# REQUIRES mcppdbg2: it reuses that patch's `_dbg_on` sampling flag, so applying
# this alone would raise NameError inside the lookup path. Checked below rather
# than left to the wrapper.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-mcppdbg3-on-cd10ed6f.patch"
readonly MARKER="${VLLM_ROOT}/.k3_mcppdbg3-915_applied"
readonly W="${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
if [[ -f "${MARKER}" ]]; then echo "[mcppdbg3-915] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[mcppdbg3-915] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if ! grep -q "_mcppdbg2_batches" "${W}"; then
  echo "[mcppdbg3-915] FATAL: mcppdbg2 is not applied; this patch reuses its _dbg_on flag." >&2
  exit 1
fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[mcppdbg3-915] content already present in this image."
else
  echo "[mcppdbg3-915] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[mcppdbg3-915] applied."
