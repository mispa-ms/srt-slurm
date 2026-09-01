#!/usr/bin/env bash
# Opt-in escape hatch: force the full-block prefix-cache path under DCP.
#
# Upstream #50493 made partial (fine-grained) prefix hits reachable when
# dcp_world_size > 1. Before it, kv_cache_coordinator.py read
#   self.enable_partial_hash_hits = dcp_world_size == 1 and has_partial_mamba_group
# so a TP8 x DCP8 prefill took the full-block path.
#
# The fine-grained finder scans hash units strictly below max_length and has no
# full-block phase, so when speculation's eagle drop lowers the ceiling by one
# hash unit the only retained Mamba boundary is stranded and the probe returns 0
# instead of degrading. On GB300 1P1D at ISL 131072 the two-wave prefix replay
# recovers 99.8% of wave 2 on the 08-19 stack and 46.9% on 08-29, which starves
# decode to a batch of one or two and makes the measurement meaningless.
#
# This does NOT remove the DCP branch. Behaviour is identical to upstream unless
# VLLM_K3_FORCE_FULL_BLOCK_PREFIX_HITS=1 is set, and the chain's #50493
# assertion still passes because the dcp_world_size > 1 test remains in place.
set -euo pipefail

readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-full-block-prefix-hits-on-6d4562c5.patch"
readonly MARKER="${VLLM_ROOT}/.k3_fullblock-829_applied"

if [[ -f "${MARKER}" ]]; then
  echo "[fullblock-829] already applied."
  exit 0
fi
if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[fullblock-829] FATAL: missing ${PATCH_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[fullblock-829] content already present in this image."
else
  echo "[fullblock-829] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  echo "[fullblock-829] a silently unpatched image measures the wrong thing." >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi

touch "${MARKER}"
echo "[fullblock-829] applied."
