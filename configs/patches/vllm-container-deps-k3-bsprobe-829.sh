#!/usr/bin/env bash
# Block-size resolution probe. Diagnostic only -- logs once, changes nothing.
#
# The two-wave prefix replay recovers 99.8% of wave 2 on the 08-19 stack and
# 46.9% here, or 0.0% with partial hash hits forced off. The finder is
# byte-identical between the stacks and MambaManager is equivalent once
# num_prefill_checkpoint_blocks is 0, so the break is in the finder's inputs.
# resolve_kv_cache_block_sizes decides those inputs and neither stack logs them.
#
# Prints one [blocksize-probe] line per engine: dcp, group count, each group's
# spec type with configured and DCP-resolved block size (and mamba cache mode),
# the resulting scheduler_block_size and hash_block_size, prefix_match_unit, and
# whether prefix caching and a connector are active.
set -euo pipefail

readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-blocksize-probe-on-6d4562c5.patch"
readonly MARKER="${VLLM_ROOT}/.k3_bsprobe-829_applied"

if [[ -f "${MARKER}" ]]; then
  echo "[bsprobe-829] already applied."
  exit 0
fi
if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[bsprobe-829] FATAL: missing ${PATCH_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[bsprobe-829] content already present in this image."
else
  echo "[bsprobe-829] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  echo "[bsprobe-829] the probe is the whole point of this arm." >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi

touch "${MARKER}"
echo "[bsprobe-829] applied."
