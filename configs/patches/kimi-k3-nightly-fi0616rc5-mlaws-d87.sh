#!/bin/bash
# kimi-k3-nightly-fi0616rc5-mlaws.sh plus a trimmed wzhao18/vllm@d87cdf5ce4.
#
# One delta against the script it wraps, so an arm on -mlaws.sh and an arm on
# this one differ by exactly this patch.
#
# WHY. Every DSpark arm on the weiport ladder died on its first request
# (pipeline 61346366, 8/8, while the no-spec twin went 8/8 green):
#
#   c16/c24/c32/c40  vllm/v1/core/kv_cache_manager.py:792, in
#                    truncate_computed_blocks
#                        assert num_blocks <= len(group_blocks)
#   c1/c2/c4/c8      CUDA error: an illegal memory access was encountered
#
# That assert is the line d87cdf5ce4 -- "Fix dspark prefix-match-unit caching
# issue" -- replaces. The trigger is not speculation in general: SpeculativeConfig
# .use_eagle() returns True for method="dspark"
#
#   return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")
#
# so the EAGLE rewind path is live for us, and with a 128-token prefix-match-unit
# on a Mamba "align" model the recurrent state the connector needs can sit a hash
# unit behind the full-attention hit. Unpatched that is an assert; patched, the
# missing positions are null placeholders and the producer caches the state at
# the reconciled replay boundary.
#
# WHAT WAS TRIMMED, AND WHY. The commit touches six files; this carries four.
# Dropped:
#   mooncake/store/coordinator.py  uses MooncakeStoreCoordinator
#                                  .eagle_attn_group_indices -- five references
#                                  in Wei's base (upstream aeeb36b1f1, 07-30),
#                                  ZERO in our nightly (cb810483, 08-04). This is
#                                  the hunk that killed all eight jobs of 61160983
#                                  and 61162407 with an AttributeError. `patch -p1`
#                                  applied it because the context lines matched; a
#                                  diff cannot see a deleted symbol.
#   mooncake/store/worker.py       docstring only.
#   vllm/envs.py                   comment only.
#
# The four that remain are a coherent set, and the assert hunk alone is NOT
# enough: kv_cache_manager stops the crash, single_type_kv_cache_manager defines
# cache_speculative_replay_tail, kv_cache_coordinator turns it on for the Mamba
# manager so the state is actually cached at that boundary, and scheduler stops
# the prefill chunk there. Applying only the first would paper over the assert
# with null blocks and is the likely source of the c1-c8 illegal access.
#
# Verified against the real nightly before this script existed: the four files
# were fetched at cb8104839c141609d99f1254459ef3a4f1bd4263, the patch applied
# with zero fuzz, all four re-parsed, and every identifier the new code uses
# resolved -- MambaSpec (import added, defined in v1/kv_cache_interface.py),
# block_pool.null_block, MambaManager, cache_speculative_replay_tail,
# speculative_replay_boundary, mamba_partial_cache_hit, hash_block_size. Zero
# occurrences of eagle_attn_group_indices remain.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5-mlaws.sh

echo "=== wzhao18/vllm@d87cdf5ce4 (trimmed to the four portable files) ==="
VLLM_DIR=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
cd "$VLLM_DIR"
patch -p1 --forward < /configs/patches/wzhao-d87cdf5ce4-trimmed.patch

python3 - <<'PY'
import pathlib, vllm
root = pathlib.Path(vllm.__file__).parent
checks = {
    "v1/core/kv_cache_manager.py": [
        ("assert num_blocks <= len(group_blocks)", False),   # the crash line is gone
        ("spec.mamba_cache_mode == \"align\"", True),
        ("self.block_pool.null_block", True),
        ("    MambaSpec,", True),
    ],
    "v1/core/kv_cache_coordinator.py": [
        ("cache_speculative_replay_tail = True", True),
        ("    MambaManager,", True),
    ],
    "v1/core/single_type_kv_cache_manager.py": [
        ("self.cache_speculative_replay_tail = False", True),
    ],
    "v1/core/sched/scheduler.py": [
        ("speculative_replay_boundary", True),
        ("if self.use_eagle and not self.mamba_partial_cache_hit:", True),
    ],
}
for rel, want in checks.items():
    src = (root / rel).read_text()
    for marker, should_exist in want:
        present = marker in src
        assert present == should_exist, f"{rel}: {marker!r} present={present}, expected {should_exist}"
    print("verified", rel)

# A patch cannot see a symbol the new code calls that upstream deleted -- that is
# exactly how d87cdf5ce4 killed eight jobs before. Prove it is not in here.
bad = [p for p in root.rglob("*.py")
       if "eagle_attn_group_indices" in p.read_text(errors="ignore")]
assert not bad, f"eagle_attn_group_indices reintroduced in {bad}"
print("no eagle_attn_group_indices anywhere in the tree")
PY
