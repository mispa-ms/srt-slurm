#!/usr/bin/env bash
# vllm#50359 -- revalidate exact partial-hash hit boundaries. The fix, not a
# diagnostic.
# =============================================================================
# WHAT IT IS. Upstream's answer to our -704, stated in its own words:
#
#   "A longer stored key proves that one object exists at that endpoint; it
#    does not imply that Mooncake also contains independently addressable
#    objects at every shorter hash boundary."
#
# Core's fine-grained lookup deliberately extends a hit into the first non-full
# block -- locally a block is usable as a prefix, remotely an object is not --
# and the EAGLE path then subtracts exactly one hash unit, landing mid-block by
# construction. Off DCP the gap is empty because the attention block equals
# hash_block_size. Scaling it by dcp opens a 1536/128 x 8 = 96-unit interior,
# and every hit landing there names a key nobody wrote.
#
# WHY IT IS NOT IN THE IMAGE. nightly 3d204dfda is 2026-08-13; the carry
# (665b7129ab) and its two guards (090492be16) are 08-10 but on a lineage the
# image never took. Six related commits are missing; these two are the ones
# that address this defect, and they touch coordinator.py only.
#
# SCOPE. Deliberately minimal. 29961d6bca (DCP + hybrid) is excluded because
# the wzhao patch already provides its effect and it touches different files;
# 0d93dac100 and dc9ae4b8ac reach into the scheduler and managers and would
# blur the A/B; 017e9f4448 is unnecessary -- the promoted argument still reads
# VLLM_PREFIX_CACHE_RETENTION_INTERVAL through get_from_deprecated_env_if_set
# and defaults to 0, which is what we set.
#
# VERIFIED, and not only textually. The patch applies onto 3d204dfda with the
# wzhao patch on top (3 hunks, one at offset 25 -- expected, since the image
# carries wzhao rather than 29961d6bca). Applying it to a tree and running the
# commits' own tests gives 41 passed, including
# test_reported_hit_is_an_object_boundary_for_every_group_under_dcp over
# dcp in {1,2,4,8} -- our case is DCP8 with multiple groups, so that test is
# the one that matters. An offset hunk that compiles is not the same as one
# that means the same thing; the tests are what close that gap.
#
# READING THE RESULT. load_get_failed_keys at c64 is 38,102 on the baseline.
# Zero means #50359 completes it. A few hundred residual means boundaries with
# nothing to step back to, and 0d93dac100 (Mamba/Eagle alignment) is next.
# Thousands means a second mechanism. Do not read "not zero" as "failed".
# =============================================================================
set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PATCH_FILE="${VLLM_PR50359_PATCH_FILE:-/configs/patches/vllm-mooncake-pr50359.patch}"
readonly MARKER_FILE="${VLLM_ROOT}/.mooncake_pr50359"

if [[ -f "${MARKER_FILE}" ]]; then
  echo "[pr50359] already applied."
  exit 0
fi

if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[pr50359] FATAL: missing patch ${PATCH_FILE}" >&2
  exit 1
fi

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "g3d204dfda" "${VERSION_FILE}"; then
  echo "[pr50359] FATAL: expected nightly g3d204dfda, got:" >&2
  cat "${VERSION_FILE}" >&2 || true
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  echo "[pr50359] content already present."
else
  echo "[pr50359] FATAL: patch neither applies nor is already applied." >&2
  exit 1
fi

python3 -m compileall -q \
  "${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py"

python3 - <<'PY'
import sys

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import coordinator as c

missing = [
    n for n, o in (
        ("tracks_existence", c.ExternalCachedBlockPool),
        ("cache_hit_alignment_tokens", c.MooncakeStoreCoordinator),
        ("_exact_partial_hit_key_exists", c.MooncakeStoreCoordinator),
    ) if not hasattr(o, n)
]
if missing:
    print("[pr50359] VERIFY FAILED, absent:", ", ".join(missing))
    sys.exit(1)
print("[pr50359] applied; exact-boundary revalidation active")
PY

touch "${MARKER_FILE}"
