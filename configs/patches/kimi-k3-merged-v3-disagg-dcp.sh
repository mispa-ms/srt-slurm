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
# wzhao18/vllm@cdcc7eae38 "fix disagg dspark", which neither our branch nor the
# v3 image carries. Its load-bearing line is in ChunkedTokenDatabase:
# speculative decoding advances token_len by up to num_spec_tokens per step
# while block_hashes covers committed tokens only, so the unhashed tail raised
# an assert out of KVCacheStoreSendingThread and killed the decode worker. The
# rest bound token ids that reach an unmasked gather -- markov_w1 (Kimi-K3's
# draft imports DSparkMarkovHead from qwen3_dspark), the penalty bin-count
# kernel, and the rejection sampler's synthetic-acceptance path, where
# draft_sampled is stored verbatim as an output id rather than compared against
# target_argmax.
WEI=/configs/patches/k3-wei-disagg-dspark.patch
echo "=== k3-wei-disagg-dspark ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$WEI" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$WEI"
fi

# wzhao18/vllm@d87cdf5ce4, the other half: fine-grained PMU under EAGLE
# drafting. EAGLE rewinds a fine-grained attention hit by one hash unit, so the
# Mamba recurrent state has to exist at that rewound boundary rather than at the
# latest prompt-tail boundary. Four pieces move together -- MambaManager caches
# there, the hybrid coordinator turns that on when eagle and partial hits are
# both live, the scheduler stops a prefill chunk there so the state is actually
# materialized, and the Mooncake coordinator resumes there on lookup. Required
# to pair DSpark with prefix-match-unit 128; without it the drafter's rewound
# hit has no recurrent state behind it.
PMU=/configs/patches/k3-wei-pmu-eagle.patch
echo "=== k3-wei-pmu-eagle ==="
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PMU" >/dev/null 2>&1; then
    echo "already applied"
else
    patch -p1 --forward -d "$SITE" < "$PMU"
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

# d87cdf5ce4's four pieces, checked where they have to agree rather than by
# their diff text. The two boundaries are computed in different files from the
# same inputs; if they drift, the state is materialized at one position and read
# at another, which is a silent Mamba miss rather than an error.
sched = read('v1/core/sched/scheduler.py')
assert 'speculative_replay_boundary' in sched, (
    'the scheduler never stops a prefill chunk at the EAGLE replay boundary, so '
    'no recurrent state is materialized there for the drafter to resume from'
)
assert 'self.use_eagle and not self.mamba_partial_cache_hit' in sched, (
    'the scheduler still backs off a full block under eagle; with a finer PMU '
    'eagle only rewinds one hash unit and the block boundary is lost'
)
mgr = read('v1/core/single_type_kv_cache_manager.py')
assert 'cache_speculative_replay_tail' in mgr, (
    'MambaManager still caches at the latest prompt hash boundary, one hash '
    'unit past where the drafter resumes'
)
coord = read('v1/core/kv_cache_coordinator.py')
assert 'cache_speculative_replay_tail' in coord, (
    'nothing turns the MambaManager flag on: the field exists but is always '
    'False, so the whole change is inert'
)
mc = read('distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py')
assert 'replay_boundary' in mc, 'the Mooncake lookup does not resume at the replay boundary'
assert 'eagle_attn_group_indices' not in mc, (
    'the source branch names the attribute eagle_attn_group_indices; here it is '
    'eagle_group_ids, and left unrenamed the fast path raises AttributeError on '
    'the first lookup'
)
guard = mc.split('replay_hit == replay_boundary')
assert len(guard) == 2, 'the replay fast path is not shaped as expected; the check below cannot read it'
assert '_exact_partial_hit_key_exists' in guard[1][:400], (
    'the replay fast path returns without revalidating that the key exists -- '
    'that is the -704 livelock of vllm#50359 reached by a new path'
)

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
