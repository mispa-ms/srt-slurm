#!/usr/bin/env bash
# Ours: drop #56104's `_recving_metadata` gate on the push producer's send
# completions. `_recving_metadata` is filled D-side only, `_sending_transfers`
# is the P side, so on a prefill worker the gate is a test against an empty dict
# and every completed push is discarded -- the request then waits out its 30 s
# lease. Same shape as #54518, which the chain already guards against in its
# earlier form. The `- failed_pushing` half of #56104 is kept.
#
# Only needed from the 09-15 nightly (cd10ed6f) onward; dc36fcce predates
# #56104, so the patch is a no-op there and this step self-skips.
set -euo pipefail
# find_spec rather than `import vllm`: importing pulls torch, which is not
# needed to place a patch and makes the step unrunnable outside a GPU container
# (tools/replay_chain.sh runs these steps to pre-flight a new nightly).
readonly VLLM_ROOT="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-pushdone-56104-on-cd10ed6f.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pushdone-915_applied"
if [[ -f "${MARKER}" ]]; then echo "[pushdone-915] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[pushdone-915] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
PW="${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py"
if ! grep -q "if req_id in self._recving_metadata" "${PW}"; then
  echo "[pushdone-915] #56104's gate is not in this image; nothing to undo."
  touch "${MARKER}"; exit 0
fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
else
  echo "[pushdone-915] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[pushdone-915] applied."
