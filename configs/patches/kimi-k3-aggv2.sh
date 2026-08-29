#!/bin/bash
# Kimi-K3 dependency setup. Shared root: ~94 configs call this, directly or
# through a wrapper, and it is the only thing they all have in common.
#
# WHAT THIS SCRIPT IS. Dependencies and verification, nothing else. It installs
# FlashInfer, the Mooncake client, ibverbs/numactl, and then checks that the
# image it landed in is the one the caller thinks it is. It applies no vLLM
# source patch of its own; callers do that after it returns.
#
# WHAT CHANGED, AND WHY THE OLD HEADER WAS WRONG. This began as the setup for a
# custom image built from mispa-ms/vllm@misunp/k3-dcp-agg-v2, and said so: "the
# image is built from ... so every vLLM source patch is already compiled in".
# That has not been true since the v5 track moved to stock nightly images plus a
# runtime patch, and it is not true for any arm running today. #50484's CUDA --
# the reason a build was needed at all -- has been upstream since 63ac04a61e, so
# every nightly from 08-11 carries torch.ops._C.direct_dcp_*, which the check at
# the bottom still asserts.
#
# THE TWO MARKER LISTS.
#   UPSTREAM  what any base image must already have. Kept current with upstream
#             moves: dcp_utils.py and the DCP half of common.py are dcp.py now.
#   OURS      what only exists once our commits are applied. A stock nightly
#             does not have them at this point, so nightly callers set
#             K3_EXPECT_OURS=0 and re-check the same markers themselves once
#             their patch is on.
#
# The OURS list still describes the pre-08-28 shape of our patch -- upstream has
# since absorbed the model_runner half -- which is correct for the older
# pinned-image callers that assert K3_EXPECT_OURS=1 and wrong for a new nightly.
# The five such callers all pin their own base SHA, so they cannot reach a
# nightly that would trip it; a new caller should set K3_EXPECT_OURS=0.
#
# FI_VER defaults to 0.6.16rc5 because three v2-era callers depend on that
# default and are pinned to images measured with it. Everything current exports
# FI_VER=0.6.16.post3 explicitly. Do not "fix" the default without re-measuring
# those three.

set -euo pipefail

FI_VER="${FI_VER:-0.6.16rc5}"
FI_CUDA="${FI_CUDA:-cu130}"

echo "=== flashinfer/vllm BEFORE ==="
python3 -m pip list 2>/dev/null | grep -iE "flashinfer|^vllm|^torch |nccl" || true

bash /configs/patches/vllm-container-deps.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl

python3 -m pip install msgpack
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 || true
python3 -m pip install --no-deps 'mooncake-transfer-engine-cuda13==0.3.12.post1'

echo "=== installing flashinfer ${FI_VER} (${FI_CUDA}) ==="
python3 -m pip install --no-deps --force-reinstall \
    "flashinfer-python==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/" \
    "flashinfer-cubin==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/${FI_CUDA}/" \
    "flashinfer-jit-cache==${FI_VER}+${FI_CUDA}" || \
    echo "WARNING: flashinfer-jit-cache ${FI_VER}+${FI_CUDA} not installed; JIT will compile on demand"

echo "=== flashinfer AFTER ==="
python3 -m pip list 2>/dev/null | grep -iE "flashinfer|^vllm" || true

# Everything below verifies the image is the one we think it is. A wrong image
# here would otherwise surface as a quiet performance difference in v2.
python3 -c "
import pathlib
import torch
import vllm

root = pathlib.Path(vllm.__file__).parent

# Two lists, because this script now has two kinds of caller.
#
# UPSTREAM: things the base image must already carry whoever calls us. #50484
# is upstream as of 63ac04a61e, so every image and every nightly from 08-11 on
# has it.
#
# OURS: things that only exist once our commits are in. An image built from one
# of our branches has them here; a stock nightly does not, and the caller
# applies them as a runtime patch AFTER this script returns. Asserting them
# unconditionally made this script fail on exactly the path it was added to
# support, so the nightly callers set K3_EXPECT_OURS=0 and check the same
# markers themselves once the patch is on.
import os

# Upstream merged v1/attention/ops/dcp_utils.py and the DCP half of
# v1/attention/ops/common.py into v1/attention/ops/dcp.py. Both markers moved
# there; checking the old paths made read_text() raise FileNotFoundError before
# any arm reached its own patch, which reads as "Server did not become healthy"
# a screen later. Checked by content, not by path, for that reason.
UPSTREAM = [
    ('models/kimi_k3/nvidia/mla.py', 'self.dcp_manager: MLADCPManager | None = None', '#50484'),
    ('v1/attention/ops/dcp.py', 'class MLADCPManager', '#50484'),
    ('v1/attention/ops/dcp.py', 'sequence_indices = torch.searchsorted(', '#50484'),
    ('envs.py', 'VLLM_USE_DIRECT_DCP_A2A', '#50484'),
]
OURS = [
    ('v1/core/kv_cache_coordinator.py',
     'dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size', '#50493'),
    ('v1/simple_kv_offload/manager.py', 'def _group_block_size', 'ours'),
    ('v1/worker/gpu/model_runner.py', 'kv_shard_count = 1 if isinstance(spec, MambaSpec)', 'ours'),
    ('models/kimi_k3/nvidia/kda_metadata.py', 'def _check_block_table_width', 'ours'),
    ('v1/core/sched/scheduler.py', 'req_hybrid_block_ids = {', 'ours'),
]
expect_ours = os.environ.get('K3_EXPECT_OURS', '1') not in ('0', '', 'false', 'False')
checks = UPSTREAM + (OURS if expect_ours else [])
if not expect_ours:
    print('K3_EXPECT_OURS=0: the caller applies our commits after this script;',
          'checking only what the base image must already have')
for rel, marker, who in checks:
    path = root / rel
    # A renamed or deleted file used to raise FileNotFoundError from read_text(),
    # which surfaces two screens later as "Server did not become healthy" and
    # tells nobody which marker moved. Upstream merged dcp_utils.py into dcp.py
    # and four GPU runs went to that. Say what is missing.
    if not path.exists():
        raise SystemExit(
            f'{who}: {rel} does not exist in this image. Upstream moved or '
            f'deleted it; update the marker list rather than the image. '
            f'(looking for: {marker})'
        )
    assert marker in path.read_text(), f'{who}: missing in {rel}: {marker}'

k3 = (root / 'models/kimi_k3/nvidia/mla.py').read_text()
assert 'does not support context parallelism.' not in k3, 'the blanket CP assert is still in mla.py'

# The reason this image exists. Without these the run is the same fallback the
# pre-v2 ladder measured, and the direct A/B arm would be measuring nothing.
missing = [n for n in ('direct_dcp_a2a_lse_reduce', 'direct_dcp_q_gather', 'direct_dcp_kv_gather')
           if not hasattr(torch.ops._C, n)]
assert not missing, (
    f'direct DCP ops missing from the image: {missing}. The CUDA kernels under '
    'csrc/libtorch_stable/attention/dcp_utils/ did not make it into the build.'
)
print('AGG v2 image verified:', root)
print('direct DCP ops present; VLLM_USE_DIRECT_DCP_* selects the path per arm')
"

# ── The Rust frontend must stay off ──────────────────────────────────────────
# It is opt-in (envs.py: VLLM_USE_RUST_FRONTEND / VLLM_USE_RUST_BENCH, both
# default "0"), and it must stay that way on this track: with it enabled the
# interactivity numbers come out wrong, and interactivity is the x axis of every
# chart in this report. A run that silently used it would not fail -- it would
# produce a plausible number on the wrong axis, which is the worst kind.
#
# Checked here rather than in any one arm's script because this file is the
# common root of every setup on the track. It guards against three things: an
# arm adding the variable to its `aggregated_environment`, a base image shipping
# it in the environment, and upstream changing the default.
python3 - <<'PY'
import os

enabled = [name for name in ("VLLM_USE_RUST_FRONTEND", "VLLM_USE_RUST_BENCH")
           if os.environ.get(name, "0") not in ("0", "", "false", "False")]
assert not enabled, (
    f"{enabled} is set. The Rust frontend distorts the interactivity metric this "
    "track plots on the x axis; unset it rather than reading the numbers."
)

# The default is what makes the check above sufficient. If upstream ever flips
# it, the variables above would be unset and the frontend still active.
import vllm.envs as envs

assert envs.VLLM_RUST_FRONTEND_PATH is None, (
    f"vLLM resolved a Rust frontend binary ({envs.VLLM_RUST_FRONTEND_PATH}) with "
    "neither opt-in variable set, so the upstream default has changed. Pin "
    "VLLM_USE_RUST_FRONTEND=0 explicitly and re-check the interactivity metric."
)
print("=== Rust frontend off (Python frontend) ===")
PY
