#!/usr/bin/env bash
# Ours: log the two factors of prefill throughput, per completed step.
#
# The 1P2D prefill saturates at ~19,600 computed tok/s on 8 GPUs (PP1) and
# ~23,200 on 16 (PP2) -- flat while the waiting queue grows 90 -> 168, so both
# are ceilings, and doubling the prefill GPUs buys 18%. Stage imbalance ([47,46]
# layers), the prefill KV pool (4-25% used), the decode as sink (same ceiling at
# 2 and 3 decode instances) and the external tier (67.7% vs 11.7% of tokens) are
# all ruled out by measurement.
#
# What is left turns on throughput = tokens-per-step x steps-per-second, and
# neither factor is logged. max_num_batched_tokens is vLLM's default 16,384
# against an ISL p50 of ~84,000, and whether a step actually carries 16,384 is
# an assumption -- with 4-7 requests running under chunked prefill it may carry
# far fewer.
#
# One line per 50th completed step:
#   [ppstep] n=N tokens=T reqs=R budget=B dt_ms=D
#
# tokens << budget            -> admission is the cap, not the budget
# tokens ~ budget, dt halves  -> the pipeline overlaps; look elsewhere
# tokens ~ budget, dt flat    -> the bubble is the answer
#
# Standalone; touches only v1/engine/core.py, which no other patch in the chain
# edits.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-ppstep-on-cd10ed6f.patch"
readonly MARKER="${VLLM_ROOT}/.k3_ppstep-915_applied"
if [[ -f "${MARKER}" ]]; then echo "[ppstep-915] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[ppstep-915] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[ppstep-915] content already present in this image."
else
  echo "[ppstep-915] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[ppstep-915] applied."
