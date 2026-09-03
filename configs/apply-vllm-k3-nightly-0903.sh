#!/usr/bin/env bash
# Kimi-K3 AgentX runtime stack on the 2026-09-03 nightly, 27a94d1c.
#
# WHY THIS REPLACES THE AUG 28 SCRIPTS. That pair carried six patches; this one
# carries a single rebased patch, because upstream merged two of them:
#
#   #53324  MooncakeStore with hybrid DCP prefix caching   merged 2026-08-29
#   #54167  low-latency GEMM fallback initialization        merged 2026-08-28
#
# and this nightly also picks up eight more that land on the bottleneck our own
# measurements pointed at -- prefix-hit retention and Mooncake accounting:
#
#   #50883  Scale UniformTypeKVCacheSpecs groups by DCP     2026-09-02
#   #54272  Fix Mooncake physical-block transfer length     2026-09-01
#   #52832  Offload producer partial tails on request finish 2026-09-01
#   #53388  Support disabling trailing prefix-cache block dropping 2026-09-01
#   #53598  Serve prefix cache hits under DCP for Kimi-K3   2026-08-31
#   #51358  Save exact Mamba boundary states                2026-08-29
#   #54697  Overlap low-M TP8 KDA projections               2026-09-02
#   #53524  Prefetch ll_bf16 router weights for M=1         2026-09-01
#
# Throughput here tracks 1/(1 - prefix_hit): +1 pp of hit is worth +27%, and the
# c48 window already oscillates between 94.25% and 96.95% instantaneous hit for
# 13,092 vs 16,809 tok/s/GPU. Anything that stops us losing hit is the lever.
#
# NOT CARRIED, on purpose. Two of the 35 k3-agent-all hunks were dropped rather
# than rebased, because these arms never reach that code:
#   simple_kv_offload/manager.py -- we run MooncakeStoreConnector and
#       SimpleCPUOffload never appears in a worker log; #50883 now owns that file.
#   worker/gpu/pp_utils.py       -- the draft discount only matters at PP > 1,
#       and these arms log pipeline_parallel_size=1.
# A PP or simple-offload arm has to put them back.
set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PATCH_FILE="${VLLM_K3_PATCH_FILE:-/configs/patches/vllm-k3-agentx-on-27a94d1c.patch}"
readonly MARKER_FILE="${VLLM_ROOT}/.k3_agentx_on_27a94d1c"
readonly LOCK="${VLLM_ROOT}/.k3-agentx.lock"

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "27a94d1c" "${VERSION_FILE}"; then
  echo "Refusing to patch: expected vLLM nightly commit 27a94d1c." >&2
  echo "Version file: ${VERSION_FILE}" >&2
  exit 1
fi
[[ -r "${PATCH_FILE}" ]] || { echo "Missing patch: ${PATCH_FILE}" >&2; exit 1; }

# All tasks on a node share the writable rootfs; serialise the apply.
exec 9> "${LOCK}"
flock -w 300 9 || { echo "timed out waiting for ${LOCK}" >&2; exit 1; }

if [[ -f "${MARKER_FILE}" ]]; then
  echo "[k3-0903] already applied"
elif patch -p1 -d "${SITE_PACKAGES}" --batch --dry-run --forward < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch -p1 -d "${SITE_PACKAGES}" --batch --forward < "${PATCH_FILE}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${MARKER_FILE}"
  echo "[k3-0903] applied"
elif patch -p1 -d "${SITE_PACKAGES}" --batch --dry-run --reverse < "${PATCH_FILE}" >/dev/null 2>&1; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "${MARKER_FILE}"
  echo "[k3-0903] content already present"
else
  echo "[k3-0903] FATAL: patch neither applies nor is present. Regenerate against this nightly." >&2
  exit 1
fi

# Assert by value. A half-applied stack serves happily and is wrong in ways
# throughput does not show, so check the things the arm actually depends on --
# including the two we now expect from upstream rather than from the patch.
python3 - <<'PY'
import importlib.util, os, sys
root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
def src(p): return open(os.path.join(root, p)).read()
fail = []
# from upstream now, not from us
if "PCP/DCP > 1 with hybrid attention" in src(
        "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py"):
    fail.append("#53324 missing: MooncakeStore still refuses hybrid DCP")
llg = src("vllm/models/kimi_k3/nvidia/low_latency_gemm.py")
i = llg.find("class _KimiK3LowLatencyApply")
if i < 0 or "super().__init__()" not in llg[i:i+600]:
    fail.append("#54167 missing: the low-latency mixin never chains __init__")
# from our patch
if "VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION" not in src("vllm/envs.py"):
    fail.append("latent-MoE tail environment gate missing")
if "VLLM_KIMI_K3_DEFERRED_MOE_FINALIZE" not in src("vllm/envs.py"):
    fail.append("deferred MoE-finalize environment gate missing")
if fail:
    sys.exit("[k3-0903] FATAL:\n  - " + "\n  - ".join(fail))
print("[k3-0903] verified: upstream #53324 + #54167 present, K3 env gates in")
PY
echo "=== k3-0903: done ==="
