#!/bin/bash
# kimi-k3-nightly-fi0616rc5-mlaws-pmu.sh plus k3-dcp-offload-hybrid-fix.patch.
#
# One delta against the script every weiport arm runs, so a pair differs by
# exactly our offload fix. No DCP, no DCP PRs: the config this serves leaves
# decode-context-parallel-size at 1.
#
# WHAT THIS IS FOR. Five SimpleCPUOffload arms on this port die at c32, all on
# the same assert, all at DCP = 1:
#
#   simpleoff        num_external_tokens=24448  not aligned to group 0 block_size=1536
#   simpleoff-safe   471424
#   so2-hma-s2c      164224
#   so2-pmu          78080
#   so2-pmu-s2c      21376
#
# Every one of those numbers is a multiple of 128 -- the configured
# prefix-match-unit -- and none is a multiple of 1536. update_state_after_alloc
# documents and asserts scheduler-block alignment while the partial hash hit
# handed to it is hash-granular. Xin Li reported this on 2026-07-29 with
# --prefix-match-unit 64 and Yifan Qiao acked it -- "I need some small tweak to
# SimpleCPUOffload to make it work with partial hit, please disable prefix match
# unit for now" -- and nothing has touched that file upstream since 2026-07-20.
#
# Our fix floors the hit at the producer, get_num_new_matched_tokens, so the
# offload tier never receives a length it cannot index. We have only ever run it
# bundled with DCP, which is not a form anyone else can use. This arm runs it on
# a stock non-DCP weiport config to show whether the fix stands on its own.
#
# Only half the patch is live here. The _group_block_size half is a no-op at
# DCP = 1, where cp_world_size is 1 and every group keeps its own block size.
# What is being tested is the one-line floor.
#
# EITHER OUTCOME IS AN ANSWER. If c32 completes, the fix unblocks a
# configuration three people are currently stuck on and can be offered as-is. If
# it still asserts, our account of the defect is wrong and needs revisiting
# before any of it goes upstream.
#
# COST OF THE FLOOR. At DCP = 1 the scheduler block is 1536 and the hash block
# is 128, so a request loses at most 1,408 tokens of CPU-tier prefix reuse --
# about 1.4% of the 104k median ISL on this workload. Watch the CPU/offload
# cache hit rate against the arm's own history, not just the throughput.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5-mlaws-pmu.sh

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")

echo "=== k3-dcp-offload-hybrid-fix (ours) ==="
FIX_FILE=/configs/patches/k3-dcp-offload-hybrid-fix.patch
if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$FIX_FILE" >/dev/null 2>&1; then
    echo "offload fix already applied"
else
    patch -p1 --forward -d "$SITE" < "$FIX_FILE"
fi

python3 -c "
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
src = (root / 'v1/simple_kv_offload/manager.py').read_text()
for marker in ('def _group_block_size',
               'hit_length = hit_length // self.block_size'):
    assert marker in src, f'offload fix missing: {marker}'

# This arm exists to test the fix without DCP. Fail closed if the DCP stack
# somehow came along, because then it is not testing what it claims to.
mla = (root / 'models/kimi_k3/nvidia/mla.py').read_text()
assert 'does not support context parallelism.' in mla, (
    'the DCP PRs are present; this arm is supposed to be stock non-DCP vLLM'
)
print('offload fix verified, DCP PRs absent, in', root)
"
