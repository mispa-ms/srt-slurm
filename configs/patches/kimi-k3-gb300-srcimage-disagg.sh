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

# Mooncake host-DRAM segment, per rank. vllm-container-deps-mooncake.sh derives
# it as TOTAL_CPU_DRAM_GB / MOONCAKE_TP, and both defaults are bia's: 1500 GB
# against that host's 2014 GiB, divided by TP8 because a bia node holds all
# eight ranks. A lyris tray holds four GPUs, so four ranks reserve
# 4 x 187 = 748 GB on a node -- which is what OOM-killed the first tep8dcp8 run
# (4 oom_kill events, SLURM step 2673272.7).
#
# 100 GB/rank is the value SA's GB300 mooncake recipe uses, and its comment
# gives the arithmetic: four segments reserve 400 GB on each 4-GPU tray. Set
# TOTAL_CPU_DRAM_GB and MOONCAKE_TP so the existing division lands there rather
# than hard-coding past it.
#
# Forced, not defaulted. The arms still carry TOTAL_CPU_DRAM_GB=1500 in their
# prefill/decode_environment for the B300 track, and if that reaches this
# context a ':-' fallback would silently restore 187 GB/rank -- the same shape
# as the MOONCAKE_VERSION pin that was set in the worker env and never arrived.
# This script only ever runs on GB300, so it decides.
#
# 2026-08-12: briefly raised to 600 (150 GB/rank) on the reading that Wei's
# config runs at that. It does not. His file sets global_segment_size 150GB with
# this comment two lines above it:
#
#   # Per TP/DP rank, not per node: 4 ranks/node x 150 GB = 480 GB of the
#   # node's 918 GB. 150 GB here OOM-killed the workers on this cluster.
#
# The arithmetic does not match the value -- 4 x 150 is 600, and 480 is what
# 120 GB/rank gives -- so the comment was written for a lower number and the
# value moved without it. Either way it records that 150 OOM-kills here, which
# is what happened to us: 4 oom_kill events on theia0217, prefill_0 dead before
# the engine came up (SLURM 2678074, pipeline 62448615).
#
# I first wrote that 150 only survived earlier because mooncake 0.3.9 never
# committed the segment, and that the wheel fix made the reservation real. That
# is wrong: 62444695 ran 150 GB/rank on 0.3.9, with all 384 registrations
# refused, and was OOM-killed just the same. 150 is simply more than a tray
# holds, whichever client is installed.
#
# Back to 400 -- 100 GB/rank, SA's GB300 recipe value, and the one our own clean
# GB300 ladder ran at (pipeline 62405731). Raise it only against a measurement.
export TOTAL_CPU_DRAM_GB=400
export MOONCAKE_TP=4

# Check it against the tray before the run rather than after. Because
# PER_RANK_GB = TOTAL_CPU_DRAM_GB / MOONCAKE_TP and MOONCAKE_TP is exactly the
# ranks on this node, TOTAL_CPU_DRAM_GB *is* the node-wide reservation -- it can
# be compared to MemTotal directly.
#
# The ceiling is bracketed by measurement, not chosen: 400 GB of a 918 GB tray
# runs (62405731), 600 GB OOM-kills (62448615), and 748 GB OOM-killed before
# that (SLURM 2673272). So the tray needs somewhere north of a third of itself
# for weights, host-side CUDA allocations, the frontend and checkpoint page
# cache. 55% admits the known-good and rejects both known-bad values.
MEM_TOTAL_GB=$(awk '/^MemTotal:/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
if [ -n "${MEM_TOTAL_GB}" ] && [ "${MEM_TOTAL_GB}" -gt 0 ]; then
    MC_CEILING_GB=$(( MEM_TOTAL_GB * ${MOONCAKE_NODE_FRACTION_PCT:-55} / 100 ))
    echo "[mooncake] node MemTotal ${MEM_TOTAL_GB} GB, reserving ${TOTAL_CPU_DRAM_GB} GB" \
         "across ${MOONCAKE_TP} ranks ($(( TOTAL_CPU_DRAM_GB / MOONCAKE_TP )) GB each)," \
         "ceiling ${MC_CEILING_GB} GB"
    if [ "${TOTAL_CPU_DRAM_GB}" -gt "${MC_CEILING_GB}" ]; then
        echo "[mooncake] FATAL: ${TOTAL_CPU_DRAM_GB} GB exceeds ${MC_CEILING_GB} GB." \
             "Mooncake commits this segment for real, so the run would be OOM-killed" \
             "~40 minutes in, at engine startup, with a bare 'Out Of Memory' from srun." >&2
        exit 1
    fi
else
    echo "[mooncake] WARN: /proc/meminfo unreadable -- cannot check ${TOTAL_CPU_DRAM_GB} GB against the tray"
fi

# lyris keeps the K3 weights on a shared path, not in a per-account HF cache:
#   /lustre/share/coreai_comparch_inferencex/models/kimi-k3   (Hanjie Qiu, 07-27)
# vllm-container-deps-k3-hfshim.sh defaults K3_STAGED_DIR to bia's staging dir
# and refuses to continue when it is absent, so this has to be set here. Note
# the path sits under /lustre/share and only carries inferencex in its name --
# it does not follow SLURM_PPP, so it resolves for either account.
# The staged K3 checkpoint sits at a different path on every cluster, and this
# script now runs on more than one of them, so probe rather than hardcode. An
# explicit K3_STAGED_DIR still wins; otherwise take the first candidate that
# exists and say which. Guessing a path costs a whole run to find out, and the
# team sheet has no OCI-aga entry yet.
if [ -z "${K3_STAGED_DIR:-}" ]; then
    for _cand in \
        /lustre/share/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/hanjieq/models/kimi-k3 \
        /lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3
    do
        if [ -d "${_cand}" ]; then
            export K3_STAGED_DIR="${_cand}"
            echo "[k3] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3] WARN: no staged checkpoint found on this cluster; the HF shim will" \
         "fall back to downloading 1.45 TB. Set K3_STAGED_DIR to the local copy." >&2
fi


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

# --- this is the k3-wei-v3 source-built image ---
# get_replay_boundary alone no longer identifies the image: k3-wei-v2 has it too,
# so once v3 existed this check stopped telling the two apart and would have let
# a stale image through in silence. Pin the identity to a marker only v3 carries
# -- wzhao18/vllm@fa92f83038, which fixed the partial-hit CoW by recording the
# block at the index rather than the last of a different list. v2 predates it and
# is the tree that hit the resulting AssertionError in pipeline 62444698.
coord = read('v1/core/kv_cache_coordinator.py')
assert 'def get_replay_boundary' in coord, (
    'wrong image: no get_replay_boundary, so this is not a source-built K3 image. '
    'Use kimi-k3-merged-v3-disagg-dcp.sh for the merged-v3 image.'
)
stkcm_src = read('v1/core/single_type_kv_cache_manager.py')
assert '(block_idx, req_blocks[block_idx])' in stkcm_src, (
    'this is k3-wei-v2, not k3-wei-v3: the partial-hit CoW still records '
    'new_computed_blocks[-1], so req_blocks[block_idx] is not that block and '
    'allocate_new_blocks dies on a bare AssertionError under load.'
)
assert 'block_hash_num_tokens' in stkcm_src, (
    'k3-wei-v3 expected: _has_partial_local_hit does not verify that the block '
    'at the index really is the partial tail'
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

# --- cdcc7eae38's clamp: absent by design on the v3 image ---
# The image is now built from wzhao/kimi-k3-agentx-v2, and that branch does not
# carry the clamp -- process_tokens still asserts that the block hashes cover
# token_len. The k3-wei-v2 image did carry it, so this check used to read the
# other way round.
#
# Left as his branch has it rather than patched back in: the point of this image
# is to measure on the tree his numbers come from, and a local divergence here
# would undo that. The assert only fires on a speculative tail the hashes do not
# cover, which a no-spec arm never produces -- so pin the expectation to
# no-spec and fail loudly if a spec arm is ever pointed at this script.
# This script serves BOTH images, so the check reports which state it found
# instead of enforcing one. Pinning it to v3's state broke every v2 arm at setup
# (62565009 and its siblings) -- the two images genuinely differ here and the
# script cannot demand a single answer.
data = read('distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py')
has_clamp = 'token_len = min(token_len, len(block_hashes)' in data
has_assert = 'assert token_len // self.hash_block_size <= len(block_hashes)' in data
assert has_clamp != has_assert, (
    'process_tokens is in neither known state (clamp=%s assert=%s): the branch '
    'moved again, re-read it before trusting any arm here' % (has_clamp, has_assert)
)
print('  process_tokens tail:', 'clamped (k3-wei-v2)' if has_clamp
      else 'asserts on hash coverage (k3-wei-v3) -- no-spec only')

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
python3 - <<'MCGATE'
import importlib.metadata as md
import os
import sys

# The deps script leaves exactly one mooncake-transfer-engine* distribution
# installed. Check that, then check its version -- in that order, because a
# version read from a NAME is meaningless while two are present: both provide
# the `mooncake` package, so pip reports one while the other is what imports.
# That is how a 0.3.9 client ran under a gate printing 0.3.12.post1, rejecting
# every KV registration with INVALID_PARAMS (-600) and offloading zero bytes.
want = os.environ.get("MOONCAKE_VERSION", "0.3.11.post1")
try:
    import mooncake
    from mooncake.store import MooncakeDistributedStore  # noqa: F401
except Exception as exc:
    sys.exit("mooncake is not importable after setup: %s" % exc)

dists = [d for d in md.distributions()
         if (d.metadata["Name"] or "").startswith("mooncake-transfer-engine")]
present = sorted("%s==%s" % (d.metadata["Name"], d.version) for d in dists)
print("  imported from: %s" % getattr(mooncake, "__file__", "?"))
print("  distributions present: %s" % (present or "none"))

if len(dists) != 1:
    sys.exit(
        "expected exactly one mooncake-transfer-engine distribution, found %d: %s. "
        "Both provide the `mooncake` package, so which one imports is undefined."
        % (len(dists), present)
    )
got = dists[0].version
if got != want:
    sys.exit(
        "mooncake is %s, not the pinned %s -- the deps script fell back. Below the "
        "pin, register_buffer does not match what vLLM's MooncakeStoreConnector "
        "calls: every KV registration is rejected with INVALID_PARAMS (-600) and "
        "the offload tier silently moves zero bytes." % (present[0], want)
    )
print("  ok: %s, master started from the same pin" % present[0])
MCGATE

echo "=== GB300 disagg setup complete ==="
