#!/usr/bin/env bash
# Ours: log the store's load decision where it is made, per request.
#
# PP2's prefill reads ~1% of what PP1 reads, with saves, lookups, key
# construction and eviction all ruled out. The decision lives in the scheduler:
# need_to_allocate is zero whenever hit_length <= num_computed_tokens, which is
# a per-request threshold rather than a difference of means. PP2's local prefix
# cache covers 82.7% against PP1's 67.1%, so more of its requests can land below
# the line. Whether that accounts for the gap is a question about a
# distribution, and every number so far is a run mean or a sample of two.
#
# Two lines per sampled request, tied by request id: the decision
# (hit_length / computed / need) and what the core scheduler granted
# (external / kvpool / vllm). Standalone -- no dependency on the earlier debug
# patches.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-mcppdbg4-on-cd10ed6f.patch"
readonly MARKER="${VLLM_ROOT}/.k3_mcppdbg4-915_applied"
readonly W="${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py"
if [[ -f "${MARKER}" ]]; then echo "[mcppdbg4-915] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[mcppdbg4-915] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[mcppdbg4-915] content already present in this image."
else
  echo "[mcppdbg4-915] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[mcppdbg4-915] applied."
