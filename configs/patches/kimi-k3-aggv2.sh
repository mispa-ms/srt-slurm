#!/bin/bash
# Kimi-K3 AGG v2 runtime setup.
#
# The image is built from mispa-ms/vllm@misunp/k3-dcp-agg-v2, so every vLLM
# source patch this track used to apply at runtime is already compiled in:
#
#   vllm main @75231eff2f          the pinned base; main is not chased
#   + vllm#50484 @65328d0ee3       Kimi-K3 DCP, INCLUDING the direct
#                                  symmetric-memory CUDA kernels under
#                                  csrc/libtorch_stable/attention/dcp_utils/
#   + vllm#50493 @9f409e3e2c       partial prefix cache hits under DCP
#   + ours                         hybrid CPU-offload block accounting
#   + ours                         stop DCP-scaling the Mamba block table
#   + ours                         hybrid-aware KV-load-failure recompute
#
# WHAT CHANGED FROM THE PATCH STACK. Two patches were dropped because upstream
# now carries them: our cudagraph empty-shard mask (superseded by 76b2e9d45e,
# same searchsorted approach) and wzhao18/vllm@d87cdf5ce4 (merged). The
# hybrid-recompute fix was a string-surgery script keyed on a source anchor; it
# is a real commit on the branch now, so an upstream edit to that block fails the
# build instead of silently not applying.
#
# WHY THE BUILD. #50484 ships 884 lines of CUDA that no pre-built nightly has, so
# every DCP number this track has published so far came off the Triton/NCCL
# fallback with VLLM_USE_DIRECT_DCP_* forced to 0 -- not the implementation the
# PR measures. The image now contains torch.ops._C.direct_dcp_*, and the ladder
# and the A/B differ only by that env var.
#
# THIS SCRIPT THEREFORE ONLY DOES DEPENDENCIES.
#
# flashinfer 0.6.16rc5 is deliberate and stays: vLLM main pins 0.6.15.post1, and
# this track has measured on rc5 throughout. Changing it here would put a second
# variable into v2.

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

for rel, marker, who in [
    ('models/kimi_k3/nvidia/mla.py', 'self.dcp_manager: MLADCPManager | None = None', '#50484'),
    ('v1/attention/ops/dcp_utils.py', 'class MLADCPManager', '#50484'),
    ('v1/attention/ops/common.py', 'sequence_indices = torch.searchsorted(', '#50484'),
    ('envs.py', 'VLLM_USE_DIRECT_DCP_A2A', '#50484'),
    ('v1/core/kv_cache_coordinator.py',
     'dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size', '#50493'),
    ('v1/simple_kv_offload/manager.py', 'def _group_block_size', 'ours'),
    ('v1/worker/gpu/model_runner.py', 'kv_shard_count = 1 if isinstance(spec, MambaSpec)', 'ours'),
    ('models/kimi_k3/nvidia/kda_metadata.py', 'def _check_block_table_width', 'ours'),
    ('v1/core/sched/scheduler.py', 'req_hybrid_block_ids = {', 'ours'),
]:
    src = (root / rel).read_text()
    assert marker in src, f'{who}: missing in {rel}: {marker}'

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
