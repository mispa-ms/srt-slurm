#!/usr/bin/env bash
# k3-merged-v3 + Mooncake-under-DCP + vllm#45340, for the DISAGGREGATED probe.
#
# ONE QUESTION: does DCP stand up across a P/D split at all. Nothing here is
# about throughput, and no arm from this script belongs on a curve yet.
#
# WHY #45340 IS NEEDED AND WHY IT IS NOT ENOUGH. MooncakeConnectorScheduler and
# NixlConnectorScheduler do their token-to-block math with the raw
# cache_config.block_size, but under DCP one block id covers block_size * dcp
# tokens, so any length that is not a multiple of the scaled block over-counts.
# #45340 (OPEN, 2026-06-12) routes both through resolve_kv_cache_block_sizes --
# the same class of fix our AGG patch made one layer down, in the store
# connector. It rebases onto our branch with two conflicts, both pure additions
# (an import beside NULL_BLOCK_ID, and speculative_config beside the new
# dcp_size/pcp_size layout keys); k3-disagg-dcp-45340.patch is the resolved form.
#
# It covers the ALIGNED case only -- prefill and decode at the same dcp. That is
# the PR's own scope, and it is why the config sets decode-context-parallel-size
# on both sides. Unaligned P/D needs remote-topology and block-position work that
# is a different PR (#38433, also open).
#
# WHAT THIS SCRIPT CANNOT TELL YOU. Whether DCP is correct across the split. The
# AGG track had the patch applying cleanly and every setup gate green for a full
# day before -704 showed the lookup was asking for keys nobody wrote. Read the
# worker log for transfer failures and the decode-side prefix cache hit rate
# before believing any number from these arms.

set -euo pipefail

# The Mooncake store half FIRST. This installs the wheel, writes
# /tmp/mooncake_config.json and launches mooncake_master -- and nothing else
# does. The disagg arms point MOONCAKE_CONFIG_PATH at that file through
# prefill/decode_environment rather than carrying a mooncake_kv_store block for
# srtslurm to render, so replacing this script with one that only patches vLLM
# leaves the path dangling and every worker dies with
#   [Errno 2] No such file or directory: '/tmp/mooncake_config.json'
# which is exactly how the first six arms of this probe failed. The aggregated
# arms hide this: there srtslurm renders the JSON from the config block, so the
# setup script never had to.
# Pin the wheel the master is launched from. This has to be exported here, not
# set in the config's prefill/decode_environment: that env reaches the vLLM
# worker, while the setup script runs earlier in a different context, so the
# value never arrived and the master came up on the 0.3.11.post1 default while
# kimi-k3-aggv2.sh later pinned the client to 0.3.12.post1. Same pattern as
# kimi-k3-merged-v3.sh exporting FI_VER.
export MOONCAKE_VERSION=0.3.12.post1

bash /configs/patches/vllm-container-deps-k3-mooncake.sh

# Then the v3 image marker, our Mooncake-under-DCP patch and its tests. Last, so
# its pins win over anything the deps scripts above reinstall.
bash /configs/patches/kimi-k3-merged-v3-mooncake.sh

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
# wzhao18/vllm@wzhao/kimi-k3-agentx-v2 -- the branch that produced the
# 1xDCP8 + 1xDCP8 / dram-mooncake curve we are reproducing. Four commits our
# tree lacked, as one patch:
#
#   cdcc7eae38  ChunkedTokenDatabase clamps token_len instead of asserting on a
#               speculative tail the block hashes do not cover, plus bounds on
#               token ids reaching an unmasked gather (markov_w1, the penalty
#               bin-count kernel, the rejection sampler).
#   5a6b8f38a9  A failed external load truncates the request at the earliest bad
#               position across groups. Our own hybrid fix reset
#               num_computed_tokens to 0 and threw away the whole prefix, so
#               under kv_load_failure_policy recompute each residual -704 cost a
#               full prompt.
#   fd3e230e7   One model-level get_replay_boundary on the coordinator, threaded
#               into every cache_blocks. Replaces d87cdf5ce4, which we carried
#               for two days before finding the chart branch does not have it --
#               Wei rewrote it three days later and this is the rewrite.
#   e4008bfc0a  effective_kv_block_size shared with resolve_kv_cache_block_sizes.
WEI=/configs/patches/k3-wei-v2.patch
echo "=== k3-wei-v2 ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$WEI" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$WEI"
fi

FIX=/configs/patches/k3-disagg-dcp-45340.patch
echo "=== k3-disagg-dcp-45340 ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$FIX" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$FIX"
fi

python3 -c "
import ast
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
read = lambda rel: (root / rel).read_text()

# Both schedulers must resolve the block size rather than read it raw. Checking
# the string alone would pass on a file that merely imports the helper, so pin
# the assignment.
for rel in ('distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py',
            'distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py'):
    src = read(rel)
    assigns = [ast.unparse(n) for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Assign)
               and 'resolve_kv_cache_block_sizes' in ast.unparse(n.value)]
    assert assigns, f'{rel} still takes its block size raw: #45340 did not apply'

# The layout key P and D compare on must carry the CP sizes, or two instances at
# different dcp would hand off KV that does not line up and never say so.
meta = read('distributed/kv_transfer/kv_connector/v1/nixl/metadata.py')
for k in ('dcp_size', 'pcp_size'):
    assert f'\'{k}\'' in meta or f'\"{k}\"' in meta, f'{k} missing from the layout key'

# Wei's clamp must be in place of the assert, or a speculative decode step whose
# tail is unhashed takes down the decode worker mid-run.
data = read('distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py')
assert 'token_len = min(token_len, len(block_hashes)' in data, (
    'wzhao18@cdcc7eae38 is missing: process_tokens still asserts on an unhashed '
    'speculative tail instead of saving the covered prefix'
)
assert 'assert token_len // self.hash_block_size <= len(block_hashes)' not in data, (
    'the old assert survives alongside the clamp'
)
rej = read('v1/worker/gpu/spec_decode/rejection_sampler_utils.py')
assert 'draft_sampled < vocab_size' in rej, (
    'the rejection sampler bounds draft ids below but not above; under synthetic '
    'acceptance an out-of-vocab id is emitted verbatim and later indexes the '
    'embedding table'
)

# fd3e230e7's pieces, checked where they have to agree rather than by diff
# text. get_replay_boundary is computed on the coordinator and consumed by every
# SingleTypeKVCacheManager; if a subclass does not accept the kwarg it is a
# TypeError on the first cached request, which a substring check cannot see.
import inspect
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core import single_type_kv_cache_manager as stkcm

assert hasattr(KVCacheCoordinator, 'get_replay_boundary'), (
    'fd3e230e7 is missing: nothing computes the model-level replay boundary'
)
missing = [
    cls.__name__
    for cls in vars(stkcm).values()
    if inspect.isclass(cls)
    and 'cache_blocks' in vars(cls)
    and 'replay_boundary' not in inspect.signature(cls.cache_blocks).parameters
]
assert not missing, (
    f'cache_blocks overrides without the replay_boundary kwarg: {missing}. The '
    f'coordinator passes it by keyword, so these raise TypeError on the first '
    f'cached request.'
)
assert not hasattr(stkcm.MambaManager, 'cache_speculative_replay_tail'), (
    'the reverted d87cdf5ce4 flag is back alongside fd3e230e7; carrying both '
    'materializes recurrent state at two different positions'
)

# A failed external load must truncate at the bad position, not discard the
# prefix. Scope this to the scanning function: _preempt_request resets
# num_computed_tokens legitimately, and on the branch we are reproducing too, so
# searching the whole file for that assignment rejects a correct tree -- which is
# exactly what it did to 62233054 and 62233056.
sched_tree = ast.parse(read('v1/core/sched/scheduler.py'))
scan = next(
    (n for n in ast.walk(sched_tree)
     if isinstance(n, ast.FunctionDef) and 'block_ids_per_group' in ast.unparse(n)),
    None,
)
assert scan is not None, (
    '5a6b8f38a9 is missing: no function scans block_ids_per_group, so the '
    'invalid-block path still unpacks a single KV group'
)
resets = [
    n for n in ast.walk(scan)
    if isinstance(n, ast.Assign)
    and ast.unparse(n) == 'request.num_computed_tokens = 0'
]
assert not resets, (
    f'{scan.name} still discards the whole prefix on a failed block '
    f'(line {resets[0].lineno}); under kv_load_failure_policy recompute every '
    f'-704 costs a full prompt'
)

from vllm.v1.core.kv_cache_utils import effective_kv_block_size  # noqa: F401

print('=== disagg DCP patch verified ===')
"

# Every assert above is a substring check, and a substring check cannot see a
# name read before it is bound. Resolving Wei's conflicts turned upstream's
# `accepted = u < rate` into `accepted &= ...` in the SYNTHETIC_MODE branch of
# _rejection_kernel; 'draft_sampled < vocab_size' was present, so the gate went
# green, and the run died twenty-five minutes later when Triton first compiled
# that specialization at warmup. Only synthetic acceptance reaches it, which is
# every throughput arm on this track.
echo "=== unbound augmented assignment ==="
python3 /configs/patches/check_unbound_augassign.py \
    "$SITE"/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py \
    "$SITE"/vllm/v1/core/sched/scheduler.py \
    "$SITE"/vllm/v1/core/kv_cache_manager.py \
    "$SITE"/vllm/v1/core/single_type_kv_cache_manager.py \
    "$SITE"/vllm/v1/core/kv_cache_coordinator.py \
    "$SITE"/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py \
    "$SITE"/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py \
    "$SITE"/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py

# The replay fast path is new code on the lookup path, so re-run the hit-boundary
# tests now that it is installed. The earlier run inside kimi-k3-merged-v3-mooncake.sh
# happens before this patch and skips the replay check for want of anything to test.
echo "=== mooncake DCP hit-boundary tests (with replay fast path) ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py

# The client the workers end up with must be the version the master was started
# from. vllm-container-deps-mooncake.sh launches the master from MOONCAKE_VERSION
# and exits; kimi-k3-aggv2.sh, later in the chain, pins the wheel to 0.3.12.post1.
# Left at the 0.3.11.post1 default those disagree, and the mismatch surfaces
# fifteen minutes in as
#   RPC call failed: invalid rpc arg
#   mount_segment_to_master_failed ... error=RPC_FAIL
#   Initialize MooncakeDistributedStore failed
# which reads like a capacity problem and is not one -- the arm that works mounts
# twice as much. Fail here instead, in a minute.
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
            f"mooncake client {got} != master {want}. The master is already "
            f"running from {want}; segment mounts will be refused with "
            f"'invalid rpc arg'. Set MOONCAKE_VERSION to the version the rest "
            f"of the chain installs."
        )
    break
else:
    sys.exit("no mooncake wheel found after setup")
print("=== mooncake versions agree ===")
PY
