#!/bin/bash
# kimi-k3-nightly-fi0616rc5-mlaws.sh plus vllm-project/vllm#51295.
#
# One delta against the script it wraps.
#
# WHAT THIS REPLACES. -mlaws-d87.sh carried a hand-trimmed four-file subset of
# wzhao18/vllm@d87cdf5ce4, a private commit. #51295 is the SAME AUTHOR'S upstream
# version of that work -- "Simplify hybrid attention eagle retention and lookup
# and fix unexpected cache miss" -- written against vLLM main rather than his
# fork, and its reproducer is our configuration verbatim:
#
#   Kimi K3, TP8 B300 / block size 1536 / prefix-match unit 128 /
#   VLLM_PREFIX_CACHE_RETENTION_INTERVAL 0
#
# Prefer this over the trimmed d87. It carries the two files we could not port
# -- mooncake/store/{coordinator,worker}.py -- because the refactor named in the
# title is what DELETED MooncakeStoreCoordinator.eagle_attn_group_indices, the
# symbol whose absence in our nightly forced the trim. The old code is replaced
# by get_prefix_replay_checkpoint / replay_alignment_tokens / eagle_replay_tokens
# in kv_cache_utils.py; nothing references the old name afterwards.
#
# WHY WE NEED IT. Under -mlaws-d87.sh the DSpark arms stopped hitting
#   kv_cache_manager.py:792  assert num_blocks <= len(group_blocks)
# and served real traffic (pipeline 61431494: c32 858 requests at 5,128
# tok/s/GPU, c16 566 at 4,463), but still lost EngineCore to a sporadic
#   CUDA error: an illegal memory access was encountered
# late in the run, taking AIPerf past its 10% error ceiling. The four files we
# had fixed the scheduler and the cache manager; the Mooncake connector kept
# doing its own un-rewound lookup. That is the gap #51295 closes.
#
# It also touches v1/simple_kv_offload/manager.py, bounding that tier's lookup
# by the same replay checkpoint -- the tier that today cannot run
# prefix-match-unit 128 above c16.
#
# TESTS ARE STRIPPED. The five tests/ files in the PR are dropped; only the
# eight vllm/ files are carried, so nothing here depends on a test tree the
# container does not ship.
#
# #51113 IS NOT STACKED. "[Bugfix] Keep mamba align prefill chunks block-aligned
# past last_cache_position" merged 2026-08-06 17:21 and rewrites the same
# _mamba_block_aligned_split. On top of this patch it applies only with fuzz 1
# and fuzz 2 -- context the patch program had to ignore, which is exactly how a
# patch lands wrong. #51295 supersedes that region; leave the rebase to upstream.
#
# VERIFIED BEFORE THIS SCRIPT EXISTED, against both container tags this arm runs:
#   cb8104839c (our pinned nightly, 08-04)  applies with ZERO fuzz
#   821717118f (latest nightly, 08-06)      applies with offsets only, no fuzz
# All eight files re-parsed and no occurrence of eagle_attn_group_indices remains.
# Our own two patchers do not collide: the hybrid-KV recompute patch edits
# scheduler.py around line 2780, this one edits 37-431.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5-mlaws.sh

echo "=== vllm-project/vllm#51295 (code files only) ==="
VLLM_DIR=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
cd "$VLLM_DIR"
patch -p1 --forward --no-backup-if-mismatch < /configs/patches/vllm-pr51295-hybrid-eagle-retention.patch

python3 - <<'PY'
import pathlib, vllm
root = pathlib.Path(vllm.__file__).parent
checks = {
    "v1/core/kv_cache_utils.py": [
        ("def get_prefix_replay_checkpoint", True),
        ("def get_prompt_cache_checkpoints", True),
    ],
    "distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py": [
        ("self.eagle_replay_tokens", True),
        ("get_prefix_replay_checkpoint", True),
    ],
    "v1/core/sched/scheduler.py": [
        ("self.replay_alignment_tokens", True),
        # our hybrid-KV recompute patch must have survived this patch
        ("req_hybrid_block_ids = {", True),
    ],
    "v1/simple_kv_offload/manager.py": [
        ("self.cpu_coordinator.get_max_cache_hit_length(request.num_tokens)", True),
    ],
    "model_executor/layers/attention/mla_attention.py": [
        # and so must the MLA chunked-prefill workspace patch
        ("tokens_per_request = cache_config.block_size", True),
    ],
}
for rel, want in checks.items():
    src = (root / rel).read_text()
    for marker, should in want:
        present = marker in src
        assert present == should, f"{rel}: {marker!r} present={present}, expected {should}"
    print("verified", rel)

# The trimmed d87 must NOT also be in the tree -- these two are alternatives.
sched = (root / "v1/core/sched/scheduler.py").read_text()
assert "speculative_replay_boundary" not in sched, \
    "the trimmed d87 patch is also applied; #51295 replaces it, do not stack them"

bad = [p for p in root.rglob("*.py")
       if "eagle_attn_group_indices" in p.read_text(errors="ignore")]
assert not bad, f"eagle_attn_group_indices still present in {bad}"
print("no eagle_attn_group_indices anywhere in the tree")
PY
