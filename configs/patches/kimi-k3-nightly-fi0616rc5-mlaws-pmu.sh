#!/bin/bash
# kimi-k3-nightly-fi0616rc5-mlaws.sh plus wzhao18/vllm@d87cdf5ce4,
# "Fix dspark prefix-match-unit caching issue".
#
# One delta against the script it wraps, so an arm on -mlaws.sh and an arm on
# this one differ by exactly that commit.
#
# WHY IT IS HERE. The commit title says dspark, and most of it is: the EAGLE
# hooks in mooncake/store/coordinator.py, kv_cache_coordinator.py,
# single_type_kv_cache_manager.py and scheduler.py all sit behind use_eagle /
# eagle_group_ids / cache_speculative_replay_tail, which a no-speculation arm
# never reaches. EAGLE-family drafting rewinds a fine-grained attention hit by
# one hash unit, and Mamba, having no draft layer of its own, still needs its
# recurrent state at that rewound boundary; that is what those hooks reconcile.
#
# One hunk is not guarded that way. kv_cache_manager.truncate_computed_blocks
# used to assert:
#
#     assert num_blocks <= len(group_blocks)
#
# and now pads with null blocks instead, under
#
#     assert isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"
#
# with the note "Mamba align lookups can lag the full-attention hit used by a
# connector". Those conditions -- a hybrid model in mamba_cache_mode align, a KV
# connector, and a prefix_match_unit finer than the block size -- are exactly the
# weiport arms: Kimi-K3, MooncakeStoreConnector, prefix-match-unit 128. Without
# this the lag trips the old assert and kills EngineCore, the same way the
# hybrid-KV unpack did in pipeline 61123650.
#
# Only the seven shipped files are carried. The commit's three test files have no
# target: vLLM is installed as a package in this image and tests/ is not part of
# it.
#
# Verified against nightly cb810483 before shipping: all seven files apply with
# no fuzz, both on the stock tree and on top of patch_kimi_k3_mooncake_hma_recompute.py.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5-mlaws.sh

echo "=== partial-prefix-hit patch (wzhao18/vllm@d87cdf5ce4) ==="
PATCH_FILE=/configs/patches/wzhao-d87cdf5ce4-partial-prefix-hits.patch
SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "partial-prefix-hit patch already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH_FILE"
fi

python3 -c "
import vllm, pathlib
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
print('partial-prefix-hit patch verified in', root)
"
