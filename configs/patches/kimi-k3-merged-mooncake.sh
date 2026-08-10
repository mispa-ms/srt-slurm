#!/usr/bin/env bash
# Merged image + Mooncake under DCP with hybrid attention.
#
# The merged image already carries #50532, DSpark-under-DCP and the flatten flag
# as commits, so kimi-k3-merged-v2.sh patches nothing and only verifies. This
# adds one runtime patch on top: MooncakeStoreConnector refused
# `kv_cache_groups > 1 and pcp * dcp > 1`, which is every DCP arm on Kimi-K3
# (24 MLA + 69 KDA groups). That is why every Mooncake arm on this track so far
# is TP8 or TP8+EP with DCP off.
#
# WHAT THE PATCH DOES. The connector indexes block_ids with GLOBAL token
# positions -- `starts // block_size` in ChunkedTokenDatabase.prepare_values --
# while the worker built one database per group from the raw spec.block_size.
# Under DCP one physical block of an ATTENTION group spans block_size * dcp
# global tokens; a Mamba group's state is replicated per rank and still spans
# block_size. resolve_kv_cache_block_sizes states exactly that rule and the
# worker already calls it for scheduler_block_size, so before this the
# coordinator was reasoning in different coordinates than the scheduler. The
# patch applies the rule per group, which is what lifts the refusal.
#
# The single-group path is deliberately untouched: it substitutes
# scheduler_block_size wholesale, which for one Mamba group is not the same
# thing, and generalising over it changed a working path to open an unrelated
# one. PCP is still refused -- different sharding, not mirrored by this scaling,
# never measured.
#
# It is Python only, so no rebuild: no CUDA, no build input. The check block
# below is the price of that, and it verifies the merged image's own markers too
# so a mispinned image fails here rather than producing a number.
#
# WHAT THIS SCRIPT CANNOT CHECK. Whether Mooncake writes a byte. Pipeline
# 61788267 ran a full Mooncake ladder in which every one of 122,449 batch_put
# calls failed with TRANSFER_FAIL and the CPU hit rate was 0.0% at all eleven
# concurrencies, and it finished green -- it measured a server with no offload
# and called it Mooncake. Read the worker log for batch_put failures and CPU hit
# rate before believing any number from these arms.

set -euo pipefail

bash /configs/patches/kimi-k3-merged-v2.sh

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

FIX=/configs/patches/k3-mooncake-dcp-hybrid.patch
echo "=== k3-mooncake-dcp-hybrid ==="
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
mc = root / 'distributed/kv_transfer/kv_connector/v1/mooncake/store'
read = lambda p: p.read_text()

# --- the refusal must no longer mention dcp ---
conn = read(mc / 'connector.py')
fn = next(n for n in ast.walk(ast.parse(conn))
          if isinstance(n, ast.FunctionDef) and n.name == '_validate_kv_cache_config')
dcp_refusals = [
    ast.unparse(n.test) for n in ast.walk(fn)
    if isinstance(n, ast.If) and 'dcp' in ast.unparse(n.test)
]
assert not dcp_refusals, f'a DCP refusal survives in the connector: {dcp_refusals}'
assert 'pcp > 1' in conn, 'the PCP refusal was dropped; it is not covered by this change'

# --- the worker must scale attention groups, and only when there are several ---
w = read(mc / 'worker.py')
tree = ast.parse(w)
scaled = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == 'block_size']
assert 'AttentionSpec' in w, 'AttentionSpec is not imported; the isinstance test cannot select groups'

# The single-group substitution must still be there, byte for byte in intent:
# one group whose spec.block_size differs from scheduler_block_size is replaced.
assert 'if len(groups) == 1:' in w, (
    'the single-group path was folded into the general rule; that changes '
    'single-Mamba + dcp>1 from scheduler_block_size to spec.block_size'
)

# And the multi-group rule must be dcp-only, matching resolve_kv_cache_block_sizes.
# The patch mirrors resolve_kv_cache_block_sizes. Check that function's CODE,
# not its prose -- an earlier version of this check matched a docstring sentence
# and broke on a line wrap while the behaviour was fine.
core_fn = next(
    n for n in ast.walk(ast.parse(read(root / 'v1/core/kv_cache_utils.py')))
    if isinstance(n, ast.FunctionDef) and n.name == 'resolve_kv_cache_block_sizes'
)
core_rule = [
    ast.unparse(n) for n in ast.walk(core_fn)
    if isinstance(n, ast.IfExp) and 'AttentionSpec' in ast.unparse(n)
]
assert core_rule and all('dcp' in r for r in core_rule), (
    'resolve_kv_cache_block_sizes no longer scales attention groups by dcp; '
    f'this patch mirrors a rule that changed: {core_rule}'
)
assert 'decode_context_parallel_size' in w

# --- vllm#50359's exact-boundary retry is now load-bearing ---
coord = read(mc / 'coordinator.py')
assert '_exact_partial_hit_key_exists' in coord, (
    'vllm#50359 is missing: without the exact-boundary retry a fine-grained hit '
    'lands inside a dcp-scaled attention block and every load of it is -704'
)
# Upstream unpacks attention_groups[0] as a 3-tuple; SpecGroup carries four
# fields here, so that raises ValueError on the first lookup. A re-import of the
# upstream form would pass `git apply` and fail at runtime -- catch it here.
fn = next(n for n in ast.walk(ast.parse(coord))
          if isinstance(n, ast.FunctionDef) and n.name == '_exact_partial_hit_key_exists')
bad = [ast.unparse(n) for n in ast.walk(fn)
       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
       and 'attention_groups' in ast.unparse(n.value)]
assert not bad, f'attention_groups[0] is tuple-unpacked, which is arity-fragile: {bad}'

print('=== Mooncake DCP-hybrid patch verified ===')
"

# The AST checks above prove the patch is present, not that it is right. Lifting
# the refusal the first time passed every one of them and then livelocked with
# 2,757,664 Mooncake OBJECT_NOT_FOUND (-704) failures -- the lookup reported a
# hit length that no object existed at. This asserts the property that failure
# violates, against the installed vllm, before any GPU time: the reported hit
# must be an exact object boundary for every group.
#
# It runs here rather than post-hoc because the accuracy gate cannot see it:
# the GSM8K arms passed at 0.950/0.954/0.956 with an external hit rate of 0.0%,
# never executing the connector's read path at all.
# Plain python, not pytest: the framework image is not guaranteed to ship it,
# and a missing test dependency must not take down every arm of a sweep.
echo "=== mooncake DCP hit-boundary tests ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py
