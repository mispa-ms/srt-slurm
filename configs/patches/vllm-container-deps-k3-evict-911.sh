#!/usr/bin/env bash
# Upstream bug, not ours: _cleanup_remote_engine asserts the engine id is in
# _remote_agents, while _evict_stale_engines drives it from _engine_last_active.
# The two maps disagree in both directions -- the function's own comment already
# tolerates a just-completed handshake with no activity entry -- and with more
# than one producer per consumer the other direction is reachable: one prefill
# idles past engine_ttl while the other serves, and the decode engine dies in
# start_load_kv -> _send_heartbeats -> _ensure_handshake (pipeline 67705738,
# both 2P1D arms). Introduced by 88ed636218 (#35264).
#
# MUTUALLY EXCLUSIVE WITH decodepp-911, which carries this same hunk plus the
# _stage_transfers cleanup. That map only exists with the decode-PP patch, so
# clearing it here would be an AttributeError on a chain without it; the chain
# refuses K3_OURS containing both.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-evict-on-e7edf17ce.patch"
readonly MARKER="${VLLM_ROOT}/.k3_evict-911_applied"
if [[ -f "${MARKER}" ]]; then echo "[evict-911] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[evict-911] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[evict-911] content already present in this image."
else
  echo "[evict-911] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[evict-911] applied."
