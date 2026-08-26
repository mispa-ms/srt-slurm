#!/usr/bin/env bash
# Kimi-K3 B200 two-node DCP8 repro of InferenceX PR #2618.
# =============================================================================
# Two things the container needs before the server starts:
#
#   1. The HF cache shim, so the ~1.4 TB checkpoint resolves to the copy already
#      staged on the cluster instead of being downloaded inside the job. On
#      prenyx that copy is under coreai_comparch_inferencex, not the bia default
#      the shim assumes, so K3_STAGED_DIR is set from the recipe environment.
#
#   2. PR #2618's dcp_local_seq_lens fix. With decode context parallelism on,
#      the MLA metadata builder swaps dcp_local_seq_lens in for seq_lens, but
#      the data-parallel idle path builds dummy input batches through
#      make_dummy(), which leaves it unset. Every DP-idle step then reads a stale
#      buffer. Upstream ships this as a patch file; it is applied here as an
#      in-place rewrite because the runtime image has neither git nor patch.
#      Source: benchmarks/multi_node/vllm_patches/kimik3-dcp-dummy-patch.sh at
#      SemiAnalysisAI/InferenceX@ad23511c.
#
# The rewrite is idempotent and asserts its anchor appears exactly once, so an
# upstream refactor fails the job here rather than silently running unpatched.
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps-k3-hfshim.sh ]]; then
    bash /configs/patches/vllm-container-deps-k3-hfshim.sh
fi

# Dump what the container can actually see of the fabric and the host. Mooncake
# names its RDMA devices by hand (device_name in the recipe) and reports a wrong
# name only as "Found 0 HCAs" / "No available RNIC", which does not say what the
# right name would have been. prenyx cost a run to that. Cheap, once per node.
echo "=== k3-b200-dcp8: fabric and host inventory ==="
echo "--- ibv_devices"; ibv_devices 2>&1 | head -20 || echo "(ibv_devices unavailable)"
echo "--- ibstat -l"; ibstat -l 2>&1 | head -20 || echo "(ibstat unavailable)"
echo "--- /sys/class/infiniband"; ls -1 /sys/class/infiniband 2>&1 | head -20 || echo "(none)"
echo "--- MemTotal"; grep -E "MemTotal|MemAvailable" /proc/meminfo || true
echo "=== k3-b200-dcp8: inventory done ==="

VLLM_ROOT=${VLLM_ROOT:-$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')}
command -v patch >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq patch; }

# ── Mooncake + DCP, vLLM PR #53324 ────────────────────────────────────────────
# Current nightlies refuse the combination outright:
#
#   MooncakeStoreConnector does not support: PCP/DCP > 1 (pcp=1, dcp=8)
#   with hybrid attention
#
# connector.py:115 on a9a17e70. The guard is not in the 728d3ad image, which is why
# every MXFP4 number on this workstream ran Mooncake with DCP8 and never met it -- and
# why dropping this PR earlier looked justified and was not. PR #53324 narrows the
# refusal to PCP alone:
#
#   -  f"PCP/DCP > 1 (pcp={pcp}, dcp={dcp}) with hybrid attention"
#   +  unsupported.append(f"PCP > 1 (pcp={pcp}) with hybrid attention")
#
# We run pcp=1, so we pass. Taken as the net diff of wzhao18/vllm:wzhao/k3-dcp-mk-2
# against main and dry-run on a9a17e70: 0 failed, 0 fuzz. The earlier attempt used the
# diff as it sat on Wei's own base and did not apply at all.
#
# Eight files, none of them model_runner.py, so this is order-independent against the
# three edits that do land there.
if grep -q "PCP/DCP > 1" "$VLLM_ROOT/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py"; then
    echo "=== mooncake-dcp: applying vLLM PR #53324 ==="
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 < /configs/patches/k3-mooncake-dcp-53324.patch > /tmp/mk53324-dry.log 2>&1; then
        echo "[mooncake-dcp] FATAL: PR #53324 does not apply to this image" >&2
        cat /tmp/mk53324-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-mooncake-dcp-53324.patch
    echo "[mooncake-dcp] PR #53324 applied; DCP no longer refused"
else
    echo "[mooncake-dcp] no PCP/DCP guard in this image; skipping PR #53324"
fi

echo "=== k3-b200-dcp8: applying the DCP dummy-batch fix ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_runner.py",
)
src = open(target).read()

marker = "input_batch.dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens"
if marker in src:
    print(f"[k3-b200-dcp8] already applied: {target}")
    sys.exit(0)

anchor = """                max_query_len=batch_desc.max_query_len,
            )
            if not skip_attn_for_dummy_run:"""
if src.count(anchor) != 1:
    sys.exit(
        f"[k3-b200-dcp8] FATAL: anchor found {src.count(anchor)} times in {target}; "
        "the dummy-batch path moved and this patch needs re-deriving"
    )

addition = """                max_query_len=batch_desc.max_query_len,
            )
            # Same DCP handling as create_dummy_attn_state: make_dummy leaves
            # dcp_local_seq_lens unset, but the MLA metadata builder swaps it
            # in for seq_lens whenever DCP is enabled.
            if self.use_dcp:
                prepare_dcp_local_seq_lens(
                    self.input_buffers.dcp_local_seq_lens,
                    input_batch.seq_lens,
                    dummy_num_reqs,
                    self.dcp_size,
                    self.dcp_rank,
                    self.cp_interleave,
                )
                input_batch.dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[
                    :dummy_num_reqs
                ]
            if not skip_attn_for_dummy_run:"""

patched = src.replace(anchor, addition)
compile(patched, target, "exec")
open(target, "w").write(patched)
print(f"[k3-b200-dcp8] applied: {target}")
PY

echo "=== k3-b200-dcp8: done ==="
