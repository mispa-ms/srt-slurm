#!/usr/bin/env bash
# Ours: on a failed lookup, say WHICH namespace answered "not present".
#
# Round one ruled out the prefix: both PP2 stages report the same groups, block
# sizes, prefix_cacheable, partial_hash_hits and hash_block_size, demand the
# right 16 prefixes, and each of the 16 ranks writes a key matching one of them.
# The conjunction at worker.py still fails, and nothing records which namespace
# said no. This logs, for the first 3 lookup batches per worker, how many of a
# group's namespaces answered yes and which ones did not -- naming the rank that
# never stored, or showing that none of them has it and the hash is the problem.
#
# Rate-limited and logging-only; a lookup batch can carry a million keys.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-mcppdbg2-on-cd10ed6f.patch"
readonly MARKER="${VLLM_ROOT}/.k3_mcppdbg2-915_applied"
if [[ -f "${MARKER}" ]]; then echo "[mcppdbg2-915] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[mcppdbg2-915] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[mcppdbg2-915] content already present in this image."
else
  echo "[mcppdbg2-915] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[mcppdbg2-915] applied."
