#!/usr/bin/env bash
# Ours: measure the pipeline bubble directly, per PP stage.
#
# ppstep splits prefill throughput into tokens-per-step x steps-per-second at
# the engine. That cannot say whether the two stages OVERLAP. Under PP the
# worker posts a non-blocking irecv and the model runner blocks on it inside
# sync_and_gather_intermediate_tensors(..., True) -- that block is stage 1
# sitting idle because stage 0 has not finished, and stage 0 never enters it.
#
# One line per 50th step per rank:
#   [ppbub] pp_rank=R n=N step_ms=S wait_ms=W wait_pct=P
#
# stage1 wait_pct high      -> the bubble is real, and P is its size
# stage1 wait_pct ~ 0       -> the stages overlap; the ceiling is upstream of
#                              the forward
# both step_ms ~ PP1's step -> the engine feeds one batch at a time; queue
#                              depth, not the stage, is the problem
#
# Touches gpu_worker.py and gpu_model_runner.py, which no patch in the 0915a
# chain edits.
set -euo pipefail
readonly VLLM_ROOT="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
readonly SITE_PACKAGES="$(dirname "${VLLM_ROOT}")"
readonly PATCH_FILE="/configs/patches/vllm-k3-ppbub-on-cd10ed6f.patch"
readonly MARKER="${VLLM_ROOT}/.k3_ppbub-915_applied"
if [[ -f "${MARKER}" ]]; then echo "[ppbub-915] already applied."; exit 0; fi
if [[ ! -r "${PATCH_FILE}" ]]; then echo "[ppbub-915] FATAL: missing ${PATCH_FILE}" >&2; exit 1; fi
if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[ppbub-915] content already present in this image."
else
  echo "[ppbub-915] FATAL: patch does not apply to this vLLM. Refusing to run:" >&2
  patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >&2 || true
  exit 1
fi
touch "${MARKER}"; echo "[ppbub-915] applied."
