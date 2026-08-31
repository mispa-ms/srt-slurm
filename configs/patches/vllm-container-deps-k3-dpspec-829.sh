#!/usr/bin/env bash
# DSpark drafting under data parallelism, for the 2026-08-29 nightly (6d4562c5).
#
# The memory-profiling path in DFlashSpeculator.propose hands the draft model
# the *target* model's dp_sync tensor, which was sized for the target's dummy
# run of max_num_batched_tokens. The draft runs num_reqs x num_query_per_req,
# so DPMetadata.make asserts two numbers from two different models --
# "AssertionError: 16384 896" on a TP1 x DP16 Kimi-K3 decode role, where
# 896 = max-num-seqs 128 x ns 7. DPMetadata is only built when DP > 1, so this
# is invisible at DP1 and fires at engine init for every wide-DP decode arm.
#
# Not fixed upstream as of main f5c3cc24 (checked 2026-08-31). The value is not
# ns-specific: at ns=4 the draft batch is 512 and still disagrees with 16384.
#
# This runs once at startup, inside memory profiling only. It cannot move a
# measured number.
set -euo pipefail

readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-dflash-dp-profile-on-6d4562c5.patch"
readonly MARKER="${VLLM_ROOT}/.k3_dpspec-829_applied"

if [[ -f "${MARKER}" ]]; then
  echo "[dpspec-829] already applied."
  exit 0
fi
if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[dpspec-829] FATAL: missing ${PATCH_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[dpspec-829] content already present in this image -- upstream landed it."
else
  echo "[dpspec-829] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  echo "[dpspec-829] a silently unpatched image fails later and less legibly." >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi

touch "${MARKER}"
echo "[dpspec-829] applied."
