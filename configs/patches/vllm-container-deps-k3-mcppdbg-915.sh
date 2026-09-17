#!/usr/bin/env bash
# Ours: log what each mooncake store rank demands at lookup and writes at save.
# Diagnostic only -- three INFO lines per rank, no behaviour change, and the
# whole block is wrapped so instrumentation cannot take down a run.
#
# For: prefill PP2 writes the store (8.1 GiB, save_exists 90% already present)
# and never reads it (load_get 0, failed_keys 0). worker.py counts a
# (group, hash) present only when EVERY namespace has it, and the namespaces
# span pp_rank in range(pp_size), so one stage chunking differently from the
# other makes the conjunction unsatisfiable with no error anywhere.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-mcppdbg-on-cd10ed6f.patch"
readonly MARKER="${VLLM_ROOT}/.k3_mcppdbg-915_applied"
if [[ -f "${MARKER}" ]]; then echo "[mcppdbg-915] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[mcppdbg-915] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[mcppdbg-915] content already present in this image."
else
  echo "[mcppdbg-915] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[mcppdbg-915] applied."
