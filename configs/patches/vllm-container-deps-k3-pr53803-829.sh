#!/usr/bin/env bash
# Upstream vllm#53803 "Fix mamba align retention checkpoints", for the
# 2026-08-29 nightly (6d4562c5). Still open upstream; it carries #53798 as its
# first commit, so applying this gives both.
#
# This replaces our own retention fix. Both target the same defect -- align mode
# materialises a mamba state block only at a chunk end, so sparse retention
# named a block that held nothing -- but from opposite ends. Ours renamed the
# block (mask the chunk end, drop the rounding); #53803 keeps the rounding grid
# and stops the align allocator from nulling those blocks out from under it,
# at the three recycling sites, plus rolling decode-end checkpoints. Upstream's
# comes with 280 lines of tests and one grid predicate shared between the mask
# and the allocator, which is the part ours never had.
#
# NOT the raw PR diff: #53803 is cut against 80771bbbd and two hunks in
# single_type_kv_cache_manager.py no longer apply to 6d4562c5, both from
# upstream refactors after that base -- _relocate_speculative_block was
# extracted into a helper, and _pending_partial_tail_offloads was renamed
# _pending_boundary_state_offloads. The shipped patch is the resolved merge.
# The keep-in-place guard is kept ahead of the helper call, which is also what
# keeps the helper's is_block_writable assert satisfied: a hashed block is
# exactly what that assert rejects.
#
# Drop this script once #53803 lands in the nightly; the guard below says so by
# refusing a forward apply that is already present.
set -euo pipefail

readonly VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-pr53803-mamba-retention-on-6d4562c5.patch"
readonly MARKER="${VLLM_ROOT}/.k3_pr53803-829_applied"

if [[ -f "${MARKER}" ]]; then
  echo "[pr53803-829] already applied."
  exit 0
fi
if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[pr53803-829] FATAL: missing ${PATCH_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pr53803-829] content already present in this image."
else
  echo "[pr53803-829] FATAL: the retention fix does not apply to this vLLM." >&2
  echo "[pr53803-829] Refusing: without it sparse retention registers grid" >&2
  echo "[pr53803-829] blocks the allocator has already nulled, and the engine" >&2
  echo "[pr53803-829] re-prefills from scratch while reporting a healthy hit." >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi

touch "${MARKER}"
echo "[pr53803-829] applied."
