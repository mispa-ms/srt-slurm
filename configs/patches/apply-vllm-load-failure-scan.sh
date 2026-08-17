#!/usr/bin/env bash
# Mooncake load-failure scan: a diagnostic, not a fix.
# =============================================================================
# The -704 failures on group:3 keys are not root-caused. Two things stand in
# the way of reading the evidence, and both are in the recv path:
#
#   worker.py -- a `break` aborts the whole load at the FIRST failing
#   sub-batch, so every later sub-batch goes unreported. Sub-batches are split
#   by staging-buffer size, not by KV cache group (every key is the same
#   24.47 MiB), so "one failure per batch, always group:3" may be the shape of
#   the defect or may be the shape of that break. Nothing in the logs tells
#   the two apart.
#
#   worker.py -- the warning prints failed[:3], so the group attribution comes
#   from a truncated sample of one sub-batch.
#
# This patch adds VLLM_MOONCAKE_STORE_LOAD_FAILURE_SCAN. When set, the loop
# attempts every sub-batch and emits one untruncated per-request summary
# counting failed keys by group:
#
#   Mooncake load failure scan: req_id=<R> failed_keys=<N> by_group={..} sub_batches=<M>
#
# Off by default: the request is already doomed at the first failure -- the
# scheduler truncates there -- so scanning trades wasted transfers for a
# complete picture and belongs only in a diagnostic arm.
#
# The same flag also re-issues batch_is_exist on the failed keys the moment the
# GET fails. -704 is OBJECT_NOT_FOUND (mooncake types.h; REPLICA_NOT_FOUND is
# -710, REPLICA_IS_GONE -712, LEASE_EXPIRED -707 -- none of which we receive),
# so asking the store about the same key now separates "it exists and the GET
# disagrees" from "the object is simply not there". The recheck is wrapped so a
# diagnostic can never change the outcome of the load it is observing.
#
# VLLM_MOONCAKE_STORE_KEY_LOG is separate because it is unbounded: one line per
# stored key, with the full key string and its put return code. The full string
# is what makes a prefix or hash-unit mismatch read as a near-miss rather than
# as absence, and the return code is what separates "never written" from
# "written and gone". c1 emits ~540 store events; do not set it above that.
#
# The scan flag also logs the group histogram of every requested key, on every
# load including successful ones. That is the denominator: without it,
# "every failure is group:3" cannot be told apart from "group:3 is most of
# what a load asks for", and the second would make the first vacuous.
#
# Pair all of it with VLLM_MOONCAKE_STORE_TIER_LOG, which prints
# memory/disk/unknown per sub-batch.
#
# Verified: the six hunks apply onto nightly 3d204dfda with Hanjie's patch
# already on top (dry-run, offsets 1/36/5/32). Unit coverage lives in
# tests/v1/kv_connector/unit/test_mooncake_store_worker.py:
#   test_recv_thread_scans_every_sub_batch_when_failure_scan_enabled
#   test_recv_thread_failure_scan_logs_untruncated_per_group_summary
# =============================================================================
set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PATCH_FILE="${VLLM_LOADSCAN_PATCH_FILE:-/configs/patches/vllm-mooncake-load-failure-scan.patch}"
readonly MARKER_FILE="${VLLM_ROOT}/.mooncake_load_failure_scan"

if [[ -f "${MARKER_FILE}" ]]; then
  echo "[loadscan] already applied."
  exit 0
fi

if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[loadscan] FATAL: missing patch ${PATCH_FILE}" >&2
  exit 1
fi

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "g3d204dfda" "${VERSION_FILE}"; then
  echo "[loadscan] FATAL: expected nightly g3d204dfda, got:" >&2
  cat "${VERSION_FILE}" >&2 || true
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  echo "[loadscan] content already present."
else
  echo "[loadscan] FATAL: patch neither applies nor is already applied." >&2
  exit 1
fi

python3 -m compileall -q \
  "${VLLM_ROOT}/envs.py" \
  "${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"

# Prove it took. A setup script that silently no-ops costs a whole run to find
# out, and this one exists only to make a log line appear.
python3 - <<'PY'
import os
import sys

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker

ok = True
for var in ("VLLM_MOONCAKE_STORE_LOAD_FAILURE_SCAN", "VLLM_MOONCAKE_STORE_KEY_LOG"):
    if not hasattr(envs, var):
        print(f"[loadscan] VERIFY FAILED: {var} not declared")
        ok = False
if not hasattr(worker, "_group_index_from_key"):
    print("[loadscan] VERIFY FAILED: _group_index_from_key missing")
    ok = False
else:
    probe = "m@tp_rank:0@pcp0@dcp0@pp_rank:0@group:3@deadbeef"
    if worker._group_index_from_key(probe) != 3:
        print("[loadscan] VERIFY FAILED: group parse wrong")
        ok = False
if not ok:
    sys.exit(1)
print("[loadscan] applied; scan=%s tier_log=%s key_log=%s" % (
    os.environ.get("VLLM_MOONCAKE_STORE_LOAD_FAILURE_SCAN", "<unset>"),
    os.environ.get("VLLM_MOONCAKE_STORE_TIER_LOG", "<unset>"),
    os.environ.get("VLLM_MOONCAKE_STORE_KEY_LOG", "<unset>"),
))
PY

touch "${MARKER_FILE}"
