#!/usr/bin/env bash
# vllm#55900 (NickLucche, opened 2026-09-08): defer recv-failure reporting until
# sibling transfers finish. Carried because it also undoes a push-producer
# regression from #54518 (merged 09-04): _pop_done_transfers reported a request
# done only `if req_id in self._recving_metadata`, which a push producer never
# has, so no WRITE ever produced done_sending and every request fell to the
# 30 s lease ("Releasing expired KV blocks ... retrieved by 0 remote worker(s)",
# 77,584 lines in pipeline 66850730, 0 on the 08-29 image). #55900 makes the
# done report unconditional and passes is_recv=False on the send side.
# MUST run after every other base_worker patch (pr50499, pushdcp, ssm): its
# hunks apply at offsets on top of them; the reverse order breaks ssm.
# Same shape as vllm-container-deps-k3-ckptidx-829.sh.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-pr55900-nixl-deferred-recv-failure-on-9ea8f3ffc.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pr55900-908_applied"
if [[ -f "${MARKER}" ]]; then echo "[pr55900-908] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[pr55900-908] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pr55900-908] content already present in this image."
else
  echo "[pr55900-908] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[pr55900-908] applied."
