#!/bin/bash
# The weiport stack with decode context parallelism, minus the MLA workspace patch.
#
#   kimi-k3-nightly-fi0616rc5-hma.sh   rc5 + the hybrid-KV recompute patch
#   + vllm#50484 / #50493 + our two fixes  (kimi-k3-nightly-fi0616rc5-dcp.sh)
#   + wzhao18/vllm@d87cdf5ce4               partial-prefix-hit fix
#
# The weiport arms normally run -mlaws-pmu.sh, which is -mlaws.sh plus the
# partial-prefix-hit fix. This script is that chain with -mlaws.sh taken out.
#
# WHY -mlaws IS OUT. patch_mla_chunked_prefill_workspace.py and #50484 rewrite
# the same two places in mla_attention.py, so they cannot both apply. Tested:
# run the patcher after #50484 and it fails closed --
#
#   RuntimeError: Expected exactly one patch target in mla_attention.py, found 0:
#   # Enforce that we enough for at least 1 page per request
#
# Dropping it costs nothing here, for two independent reasons:
#
#   1. The arms this script serves pin max-num-seqs to 2*CONC. The floor the
#      patcher removes only bites at max_num_seqs >= 43, which the
#      -weiport-simpleoff-safe- config says outright: "makes the MLA workspace
#      patch a no-op here".
#   2. #50484 already does the patcher's job. Both keep a page-per-request floor
#      when DCP is on and relax it otherwise -- Wei's as
#      align_chunk_to_block=self.dcp_world_size > 1, #50484's as
#      align_mla_chunked_context_workspace_size, which additionally rounds to
#      lcm(block_size, dcp * cp_kv_cache_interleave_size). #50484 is the superset.
#
# So this script is only valid for arms with max-num-seqs pinned. An arm that
# leaves max-num-seqs at Wei's default of 1024 would get a
# 1024 * 1536-row workspace in the model dtype, which is why the frontier arm
# is not reachable this way.
#
# ORDER. The DCP stack goes on before the partial-prefix-hit fix. Both touch
# kv_cache_coordinator.py and kv_cache_manager.py; verified in that order
# against a pristine cb810483 tree, where the fix applies with line offsets only
# and no fuzz.
#
# WHAT IS STILL UNKNOWN HERE. The arms carry prefix-match-unit 128, and
# k3-dcp-offload-hybrid-fix.patch re-aligns CPU-tier hits to the scheduler
# block, which under DCP4 is 1536 * 4 = 6144. So the CPU tier matches 48x
# coarser than the config asks for. It did not matter on the -cur- ladder, where
# the DCP arm used the CPU tier for 0.5% of its hits, but these arms lean on it
# much harder. Read the CPU/offload cache hit rate before the throughput.

set -euo pipefail

DCP_BASE_SCRIPT=/configs/patches/kimi-k3-nightly-fi0616rc5-hma.sh \
    bash /configs/patches/kimi-k3-nightly-fi0616rc5-dcp.sh

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")

echo "=== partial-prefix-hit patch (wzhao18/vllm@d87cdf5ce4) ==="
PATCH_FILE=/configs/patches/wzhao-d87cdf5ce4-partial-prefix-hits.patch
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "partial-prefix-hit patch already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH_FILE"
fi

python3 -c "
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
checks = {
    'v1/core/kv_cache_manager.py': 'spec.mamba_cache_mode == \"align\"',
    'v1/core/kv_cache_coordinator.py': 'manager.cache_speculative_replay_tail = True',
    'v1/core/single_type_kv_cache_manager.py': 'self.cache_speculative_replay_tail',
    'v1/core/sched/scheduler.py': 'speculative_replay_boundary',
    'distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py': 'replay_boundary',
}
for rel, marker in checks.items():
    src = (root / rel).read_text()
    assert marker in src, f'partial-prefix-hit patch missing in {rel}: {marker}'

# The DCP stack has to have survived the fix landing on the same two files.
coord = (root / 'v1/core/kv_cache_coordinator.py').read_text()
assert 'dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size' in coord, (
    'the partial-prefix-hit patch displaced #50493 in kv_cache_coordinator.py'
)
print('weiport + DCP stack verified in', root)
"
