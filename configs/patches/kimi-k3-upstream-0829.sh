#!/usr/bin/env bash
# The 08/29 nightly (6d4562c59b) with the upstream-only chain. Container bisect.
# =============================================================================
# WHY THIS IMAGE AND NOT 08/28. The first two attempts at this experiment used
# 6f7df92a8e and both died in setup, for the same reason twice: that image lacks
# things the 08/31 image has built in, so a chain built by asking only "which of
# the newer patches still apply" is incomplete.
#
#   attempt 1 (65686823)  AttributeError: 'KimiK3LowLatencyLinearMethod' object
#                         has no attribute '_gemm_impl'   -- #54167 merged
#                         08-28 08:32, that nightly was cut 06:13.
#   attempt 2 (65689710)  ValueError: MooncakeStoreConnector does not support:
#                         PCP/DCP > 1 (pcp=1, dcp=8) with hybrid attention
#                         -- #53324 (the refusal lift) merged 08-29.
#
# The second one is not a missing patch, it is a dead end: to boot Mooncake +
# DCP8 + hybrid on 08/28 you MUST carry our connector hunk, and for that hunk to
# be sound you must also carry the worker's per-group scaling. "Our chain without
# the Mooncake carry, on 08/28" is therefore not constructible -- most of the
# carry has to come back before the engine starts, at which point it is not the
# chain being tested. The experiment was ill-posed, not merely under-patched.
#
# 08/29 is the earliest container where the upstream-only chain can boot at all,
# because it is the first with #53324.
#
# WHAT THIS IMAGE ALREADY HAS, verified against the tree, not assumed:
#   #53324  refusal lifted -- "PCP/DCP > 1" is absent from connector.py
#   #54167  gemm super().__init__() present -- k3-gemmfix does not apply, and
#           must not: it is already there
#   #51358  _apply_current_save_block_ids present -- so our save-snapshot patch
#           is both needed and applicable here
#
# WHAT IT LACKS, so the question never arises:
#   #50611  no interleave promotion -- upstream-pr54457 does not apply and there
#           is nothing for our MooncakeStoreConnector opt-out to gate
#   #54044  no mamba align-metadata reset (we carry no mambacache patch anyway)
#
# So the chain here is #50514 + int64idx + save-snapshot, which is semantically
# the same content as the 08/31 v2 chain: the two patches left out are inert on
# this image. Container is the only thing that moves.
#
# WHAT THE RESULT MEANS. Compare external prefix-cache hit against the mcv8 arm
# of the same name, which scored 83.0% on 08/28 with zero store-put failures,
# and against our 08/31 arm, which scored 0.0% with 60,152 puts failing -900.
#
#   near 83%   ->  the regression is in 08/29..08/31, a much smaller window
#   still 0%   ->  it is in 08/28..08/29 -- which contains BOTH #53324 and
#                  #51358, i.e. the container change and our carry removal are
#                  the same commits, the two were never separable, and the
#                  investigation goes straight at those two PRs
#
# Read external_cache_hit and the -900 count. Not tok/s: c48 is past the ns7
# cliff, which peaks at c32.
# =============================================================================
set -euo pipefail

readonly PINNED_SHA=6d4562c59b
readonly PATCH_DIR=/configs/patches

echo "=== kimi-k3-upstream-0829: container bisect, on ${PINNED_SHA} ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
echo "[upstream-0829] vllm root: ${VLLM_ROOT}"

python3 - <<PY
import re
import vllm
m = re.search(r"\+g([0-9a-f]+)", vllm.__version__)
sha = m.group(1) if m else "?"
pinned = "${PINNED_SHA}"
if sha.startswith(pinned) or pinned.startswith(sha):
    print(f"[upstream-0829] base image confirmed: {sha}")
else:
    raise SystemExit(
        f"[upstream-0829] FATAL: base is {sha}, this chain is only valid on {pinned}. "
        "It exists to bisect the container against the 08/31 arms; anywhere else it "
        "answers nothing."
    )
PY

apply_patch() {
    local name="$1" marker="$2" file="$3"
    if grep -q "${marker}" "${VLLM_ROOT}/${file}" 2>/dev/null; then
        echo "[upstream-0829] ${name}: already present, skipping"
        return 0
    fi
    if ! patch -p1 -d "${VLLM_ROOT}" --dry-run --forward --fuzz=0 \
         < "${PATCH_DIR}/${name}" > "${TMPDIR:-/tmp}/${name}.dry" 2>&1; then
        echo "[upstream-0829] FATAL: ${name} does not apply" >&2
        cat "${TMPDIR:-/tmp}/${name}.dry" >&2
        exit 1
    fi
    patch -p1 -d "${VLLM_ROOT}" --forward --fuzz=0 < "${PATCH_DIR}/${name}"
    echo "[upstream-0829] ${name}: applied"
}

apply_patch upstream-pr50514.patch \
    'supports_aux_hidden_states_over_pp' \
    vllm/models/kimi_k3/nvidia/model.py

apply_patch ours-k3-int64idx.patch \
    'seq_idx).to(tl.int64)' \
    vllm/models/kimi_k3/nvidia/kda.py

apply_patch ours-mooncake-save-snapshot.patch \
    'no current block table for' \
    vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py

# Assert both directions: what we added, and what this image must already have.
# Attempt 1 and 2 both died on the second kind, which nothing was checking.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
rd = lambda p: open(os.path.join(root, p)).read()
MC = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py"
CN = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py"
failures = []

def want(rel, needle, why):
    if needle not in rd(rel):
        failures.append(why)

def forbid(rel, needle, why):
    if needle in rd(rel):
        failures.append(why)

# what our patches added
forbid("vllm/v1/worker/gpu/model_runner.py", "with pipeline parallel ",
       "PP+spec refusal was not lifted")
want("vllm/models/kimi_k3/nvidia/model.py",
     "supports_aux_hidden_states_over_pp = True",
     "K3 does not opt in to aux hidden states over PP")
want("vllm/models/kimi_k3/nvidia/kda.py", "seq_idx).to(tl.int64)",
     "the KDA checkpoint state index is still int32")
forbid(MC, "Missing current block table for store request",
       "the fatal Mooncake save assert is still present")
want(MC, "no current block table for",
     "the Mooncake save is not declined when the core has no snapshot")

# what this image must already provide -- the class of failure that killed
# attempts 1 and 2 on the 08/28 image
want("vllm/models/kimi_k3/nvidia/low_latency_gemm.py", "super().__init__()",
     "#54167 is missing: profile_run will die on _gemm_impl")
forbid(CN, "PCP/DCP > 1",
       "#53324 is missing: the connector will refuse DCP8 + hybrid at init")

for f in failures:
    print(f"[upstream-0829] FATAL: {f}", file=sys.stderr)
sys.exit(1 if failures else 0)
PY

echo "[upstream-0829] all 7 assertions passed"

python3 - <<'PY'
import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler  # noqa: F401
import vllm.v1.worker.gpu.model_runner  # noqa: F401

print("[upstream-0829] edited modules import cleanly")
PY

echo "=== kimi-k3-upstream-0829: done ==="
