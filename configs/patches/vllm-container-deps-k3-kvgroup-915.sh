#!/usr/bin/env bash
# Ours: choose the KV-cache group size by minimising padding, behind a flag.
#
# The GB300 disagg prefill logs "Add 18 padding layers, may waste at most
# 26.09% KV cache memory". That is the coverage lever: y tracks
# delivered = computed / (1 - coverage) (r = +0.69..+0.995 within a ladder),
# coverage is set by the prefill's HBM prefix cache, and that cache collapses
# from 72.3% hit at c48 to 3.9% at c144. PP2's only real advantage is a 5.3x
# larger pool; buying that with eight more GPUs is what keeps it off the
# frontier, and buying it back from padding is free.
#
# kv_cache_utils.py picks group_size as min(bucket sizes), or max when they are
# within 1.5x. K3 is 69 KDA + 24 MLA and the DSpark drafter adds 5 MLA layers,
# so the buckets are 69 and 29 -> group_size 29 -> KDA pads 69 to 87. The
# function carries a FIXME saying the heuristic is wrong for exactly this shape.
#
# With VLLM_KV_GROUP_MIN_PADDING=1 it picks the group size minimising total
# padding under VLLM_KV_GROUP_MAX_GROUPS (default 16): for K3, 10 instead of 29,
# padding 18 -> 2 layers, groups 4 -> 10. Unset, behaviour is byte-identical --
# the group count feeds the Mooncake key namespaces and the NIXL region layout,
# so both halves of a disagg pair must set it or neither.
#
# Patch is applied to a throwaway tree, byte-compiled, and the decision logic
# exercised on K3's real bucket sizes before submission.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-kvgroup-on-cd10ed6f.patch"
readonly MARKER="${VLLM_ROOT}/.k3_kvgroup-915_applied"
if [[ -f "${MARKER}" ]]; then echo "[kvgroup-915] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[kvgroup-915] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[kvgroup-915] content already present in this image."
else
  echo "[kvgroup-915] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[kvgroup-915] applied."
