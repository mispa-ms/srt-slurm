#!/usr/bin/env bash
# Kimi-K3 DISAGG on an image source-built from misunp/k3-wei-v2 (GB300 / lyris).
# =============================================================================
# The B300 chain exists to turn the k3-merged-v3 image into the tree we want, by
# applying k3-mooncake-dcp-hybrid.patch and k3-wei-v2.patch at runtime and
# asserting at each step that it started from the image it expects. The aarch64
# image is built from that tree directly, so every one of those patches is
# already in it and every one of those identity checks is asking the wrong
# question -- three of them rejected it outright and cost 23 jobs.
#
# So this does not patch vLLM. It installs what the image cannot carry, applies
# the one patch that is genuinely absent, and verifies the tree it actually has.
#
# WHAT IS STILL NEEDED, and why the chain cannot simply be skipped:
#   HF shim         the model cache has to be wired to the pre-staged checkpoint
#   Mooncake        wheel, /tmp/mooncake_config.json, and the master process --
#                   nothing else starts it, and the disagg arms point
#                   MOONCAKE_CONFIG_PATH at that file
#   flashinfer      pinned per-arm, not baked into the image
#   ibverbs/numactl apt packages the runtime needs
#   vllm#45340      the only vLLM patch NOT in the image: it is applied at
#                   runtime on both tracks and was never part of k3-wei-v2
#
# WHAT IS DELIBERATELY NOT DONE: k3-wei-v2.patch and k3-mooncake-dcp-hybrid.patch.
# Both are in the image. Re-applying is a no-op the chain detects, but the
# identity assertions guarding them are not.
# =============================================================================
set -euo pipefail

export MOONCAKE_VERSION=0.3.12.post1
export FI_VER=${FI_VER:-0.6.16.post3}

# hfshim first -- the model has to resolve before anything else matters -- then
# the Mooncake wheel, config and master. Both scripts source
# vllm-container-deps.sh themselves; apt/pip are idempotent.
bash /configs/patches/vllm-container-deps-k3-mooncake.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl

FI_CUDA=$(python3 -c "
import torch
major = torch.version.cuda.split('.')[0]
print(f'cu{major}' + ('0' if major == '13' else '8'))
")
echo "=== installing flashinfer ${FI_VER} (${FI_CUDA}) ==="
python3 -m pip install --no-deps --force-reinstall "flashinfer-python==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/" "flashinfer-cubin==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/${FI_CUDA}/" \
    "flashinfer-jit-cache==${FI_VER}+${FI_CUDA}" || \
    echo "WARNING: flashinfer-jit-cache ${FI_VER}+${FI_CUDA} not installed; JIT will compile on demand"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

# The one vLLM patch the image does not have. #45340 routes the Mooncake and
# NIXL schedulers through resolve_kv_cache_block_sizes instead of the raw
# cache_config.block_size, which under DCP over-counts any length that is not a
# multiple of the scaled block.
FIX=/configs/patches/k3-disagg-dcp-45340.patch
echo "=== k3-disagg-dcp-45340 ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$FIX" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$FIX"
fi

# Verify the tree this image actually has, rather than the one the B300 chain
# builds. Every check below is on code, not on diff text: a substring check
# passes on a file where the symbol is present but unreachable, which is how the
# accepted &= defect and the enable_partial_hash_hits ordering both got through.
python3 -c "
import ast
import inspect
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
read = lambda rel: (root / rel).read_text()

# --- this is the source-built image, not k3-merged-v2 or v3 ---
coord = read('v1/core/kv_cache_coordinator.py')
assert 'def get_replay_boundary' in coord, (
    'wrong image: no get_replay_boundary, so this is not built from '
    'misunp/k3-wei-v2. Use kimi-k3-merged-v3-disagg-dcp.sh for the v3 image.'
)

# --- fd3e230e7: one model-level boundary, threaded as a required kwarg ---
from vllm.v1.core import single_type_kv_cache_manager as stkcm
missing = [
    cls.__name__
    for cls in vars(stkcm).values()
    if inspect.isclass(cls) and 'cache_blocks' in vars(cls)
    and 'replay_boundary' not in inspect.signature(cls.cache_blocks).parameters
]
assert not missing, (
    f'cache_blocks overrides without the replay_boundary kwarg: {missing}. The '
    f'coordinator passes it by keyword, so these raise TypeError on the first '
    f'cached request.'
)
assert not hasattr(stkcm.MambaManager, 'cache_speculative_replay_tail'), (
    'the reverted d87cdf5ce4 flag is present alongside fd3e230e7; carrying both '
    'materializes recurrent state at two different positions'
)

# --- 5a6b8f38a9: truncate at the failed block, not the whole prefix ---
scan = next(
    (n for n in ast.walk(ast.parse(read('v1/core/sched/scheduler.py')))
     if isinstance(n, ast.FunctionDef) and 'block_ids_per_group' in ast.unparse(n)),
    None,
)
assert scan is not None, (
    '5a6b8f38a9 is missing: the invalid-block path still unpacks a single group'
)
resets = [
    n for n in ast.walk(scan)
    if isinstance(n, ast.Assign) and ast.unparse(n) == 'request.num_computed_tokens = 0'
]
assert not resets, (
    f'{scan.name} still discards the whole prefix on a failed block '
    f'(line {resets[0].lineno})'
)

# --- cdcc7eae38: clamp instead of asserting on an unhashed speculative tail ---
data = read('distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py')
assert 'token_len = min(token_len, len(block_hashes)' in data, (
    'process_tokens still asserts on a speculative tail the block hashes do not '
    'cover, which kills the decode worker mid-run'
)
assert 'assert token_len // self.hash_block_size <= len(block_hashes)' not in data, (
    'the old assert survives alongside the clamp'
)

# --- e4008bfc0a: the DCP scaling rule, shared ---
from vllm.v1.core.kv_cache_utils import effective_kv_block_size  # noqa: F401
conn = read('distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py')
fn = next(n for n in ast.walk(ast.parse(conn))
          if isinstance(n, ast.FunctionDef) and n.name == '_validate_kv_cache_config')
dcp_refusals = [
    ast.unparse(n.test) for n in ast.walk(fn)
    if isinstance(n, ast.If) and 'dcp' in ast.unparse(n.test)
]
assert not dcp_refusals, f'a DCP refusal survives in the connector: {dcp_refusals}'
assert 'pcp > 1' in conn, 'the PCP refusal was dropped; it is not covered here'

# --- vllm#45340, applied just above ---
for rel in ('distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py',
            'distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py'):
    src = read(rel)
    assigns = [ast.unparse(n) for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Assign)
               and 'resolve_kv_cache_block_sizes' in ast.unparse(n.value)]
    assert assigns, f'{rel} still takes its block size raw: #45340 did not apply'
meta = read('distributed/kv_transfer/kv_connector/v1/nixl/metadata.py')
for k in ('dcp_size', 'pcp_size'):
    assert f'\'{k}\'' in meta or f'\"{k}\"' in meta, f'{k} missing from the layout key'

print('=== GB300 source-built image verified ===')
"

echo "=== unbound augmented assignment ==="
python3 /configs/patches/check_unbound_augassign.py \
    "$SITE"/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py \
    "$SITE"/vllm/v1/core/sched/scheduler.py \
    "$SITE"/vllm/v1/core/kv_cache_coordinator.py \
    "$SITE"/vllm/v1/core/single_type_kv_cache_manager.py \
    "$SITE"/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py \
    "$SITE"/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py

echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py

# The client the workers end up with must be the version the master was started
# from; a mismatch surfaces fifteen minutes in as 'invalid rpc arg' and reads
# like a capacity problem.
echo "=== mooncake client/master version agreement ==="
python3 - <<'PY'
import importlib.metadata as md
import os
import sys

want = os.environ.get("MOONCAKE_VERSION", "0.3.11.post1")
for pkg in ("mooncake-transfer-engine-cuda13", "mooncake-transfer-engine"):
    try:
        got = md.version(pkg)
    except md.PackageNotFoundError:
        continue
    print(f"  {pkg}: installed {got}, master started from {want}")
    if got != want:
        sys.exit(
            f"mooncake client {got} != master {want}. Segment mounts will be "
            f"refused with 'invalid rpc arg'."
        )
    break
else:
    sys.exit("no mooncake wheel found after setup")
PY

echo "=== GB300 disagg setup complete ==="
