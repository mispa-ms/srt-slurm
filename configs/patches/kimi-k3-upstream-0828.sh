#!/usr/bin/env bash
# The 08/28 nightly (6f7df92a8e) with the upstream-only content that applies there.
# =============================================================================
# WHY THIS EXISTS. It isolates one variable. The mcv8 arms on this image reached
# 83.0% external prefix-cache hit with zero store-put failures. Our upstream-only
# arms on the 08/31 image reached 0.0% with 57,000-60,000 puts failing
# `-900 invalid rpc arg` from the Mooncake master, and lost 31-64% throughput.
# Two things moved between those runs -- the container AND the chain -- so neither
# has been shown to be the cause.
#
# This chain puts our content on the OLD container. Then:
#
#   external hit back near 83%  ->  the CONTAINER is the culprit; the search
#                                   narrows to the 87 commits 6f7df92a8e..44fe2a392b
#   external hit still 0.0%     ->  DROPPING OUR MOONCAKE CARRY is the culprit,
#                                   and k3-ours6-v8-nightly.patch's three files
#                                   need a line-by-line diff against upstream
#
# WHAT IT CARRIES, AND WHY IT IS NOT THE v1/v2 LIST. Two of those five patches
# target code that does not exist on this image, verified by dry-run:
#
#   upstream-pr54457.patch        FAILS -- patches adjust_dcp_kv_cache_interleave_size,
#                                 added by #50611 on 08-29. Moot here: there is no
#                                 interleave promotion on this image to gate.
#   ours-mooncake-save-snapshot   FAILS -- patches _apply_current_save_block_ids,
#                                 added by #51358 on 08-29. Moot here: the assert
#                                 that kills EngineCore does not exist yet.
#
# So the meaningful content on this image is #50514 plus int64idx, and that is
# exactly what makes this a clean discriminator: the only thing separating this
# arm from the mcv8 arm that scored 83% is our Mooncake/mambacache carry.
#
#
# WHAT THIS IMAGE NEEDS THAT THE NEWER ONE DOES NOT. The first attempt at this arm
# died in profile_run with
#
#   AttributeError: 'KimiK3LowLatencyLinearMethod' object has no attribute '_gemm_impl'
#
# because vllm#54167 (the missing super().__init__() in low_latency_gemm.py) merged
# 2026-08-28 08:32 and this nightly was cut at 06:13 -- two hours short. The 08/31
# image has it built in, which is why the v1/v2 chains correctly drop our carry.
# Going backwards to this container means carrying it again. Asking only "which of
# the newer patches still apply" was not enough; an older base can also need
# something the newer one no longer does.
#
# ours-mooncake-interleave.patch is deliberately NOT applied. It applies cleanly
# but is inert -- nothing on this image reads
# requires_dcp_block_aligned_interleave -- and leaving it out keeps the arm's
# content exactly "what upstream has, plus the one line no PR covers".
# =============================================================================
set -euo pipefail

readonly PINNED_SHA=6f7df92a8e
readonly PATCH_DIR=/configs/patches

echo "=== kimi-k3-upstream-0828: isolating container vs chain, on ${PINNED_SHA} ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[upstream-0828] vllm root: ${VLLM_ROOT}"

python3 - <<PY
import re
import vllm
m = re.search(r"\+g([0-9a-f]+)", vllm.__version__)
sha = m.group(1) if m else "?"
pinned = "${PINNED_SHA}"
if sha.startswith(pinned) or pinned.startswith(sha):
    print(f"[upstream-0828] base image confirmed: {sha}")
else:
    raise SystemExit(
        f"[upstream-0828] FATAL: base is {sha}, this chain is only valid on {pinned}. "
        "It exists to compare against the 08/31 arms on the OLD container; running it "
        "anywhere else answers nothing."
    )
PY

apply_patch() {
    local name="$1" marker="$2" file="$3"
    if grep -q "${marker}" "${VLLM_ROOT}/${file}" 2>/dev/null; then
        echo "[upstream-0828] ${name}: already present, skipping"
        return 0
    fi
    if ! patch -p1 -d "${VLLM_ROOT}" --dry-run --forward --fuzz=0 \
         < "${PATCH_DIR}/${name}" > "${TMPDIR:-/tmp}/${name}.dry" 2>&1; then
        echo "[upstream-0828] FATAL: ${name} does not apply" >&2
        cat "${TMPDIR:-/tmp}/${name}.dry" >&2
        exit 1
    fi
    patch -p1 -d "${VLLM_ROOT}" --forward --fuzz=0 < "${PATCH_DIR}/${name}"
    echo "[upstream-0828] ${name}: applied"
}

apply_patch k3-gemmfix-54167.patch \
    'super().__init__()' \
    vllm/models/kimi_k3/nvidia/low_latency_gemm.py

apply_patch upstream-pr50514.patch \
    'supports_aux_hidden_states_over_pp' \
    vllm/models/kimi_k3/nvidia/model.py

apply_patch ours-k3-int64idx.patch \
    'seq_idx).to(tl.int64)' \
    vllm/models/kimi_k3/nvidia/kda.py

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
rd = lambda p: open(os.path.join(root, p)).read()
fail = []

if "with pipeline parallel " in rd("vllm/v1/worker/gpu/model_runner.py"):
    fail.append("PP+spec refusal was not lifted")
if "supports_aux_hidden_states_over_pp = True" not in rd("vllm/models/kimi_k3/nvidia/model.py"):
    fail.append("K3 does not opt in to aux hidden states over PP")
if "seq_idx).to(tl.int64)" not in rd("vllm/models/kimi_k3/nvidia/kda.py"):
    fail.append("the KDA checkpoint state index is still int32")
if "super().__init__()" not in rd("vllm/models/kimi_k3/nvidia/low_latency_gemm.py"):
    fail.append("#54167 is missing: profile_run will die on _gemm_impl")

# This image must NOT have the two things the newer chain works around. If it
# does, the image is not what we think it is and the comparison is void.
if "_apply_current_save_block_ids" in rd(
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py"
):
    fail.append("#51358 is present: this is not the pre-51358 image")

for f in fail:
    print(f"[upstream-0828] FATAL: {f}", file=sys.stderr)
sys.exit(1 if fail else 0)
PY

echo "[upstream-0828] all 4 assertions passed"
echo "=== kimi-k3-upstream-0828: done ==="
