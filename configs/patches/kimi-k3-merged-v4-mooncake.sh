#!/usr/bin/env bash
# k3-merged-v4 + Mooncake under DCP with hybrid attention.
#
# NOTHING IS PATCHED HERE. On v3 this was a runtime patch, which is why those
# nineteen Mooncake runs could start the moment the branch was pushed and never
# waited for an image. v4 is a rebase, so the same change rides in as commits
# 8d6daca04f / 9cc5117996 / f4f10fbcca and the image already carries it. What is
# left is the verification, and it matters more here than it did on v3: with
# nothing to apply, a patch that stopped applying can no longer fail loudly, and
# the only remaining failure mode is a mispinned image producing a number.
#
# WHAT v4 CHANGES UNDER THIS ARM. #51766 (merged 2026-08-11) preserves Mamba
# running CoW after external hits. Every Mooncake load on Kimi-K3 crosses 69 KDA
# groups, so this is the read path these arms exercise and not an adjacent one;
# it is the reason to expect a v3->v4 delta here rather than noise. #51749
# generalises KV block zeroing to AttentionSpec, which is the same per-group
# distinction the DCP block-size rule below rests on.
#
# #50344's gate is still in force: MooncakeStoreConnector does not declare
# supports_divergent_local_hybrid_hits, so it takes the conservative
# common-prefix path. The comparison is against the v3 Mooncake ladder, which
# took that same path -- 12,630 tok/s/GPU at c70, and a DSpark cliff of -19.9%
# rather than SimpleCPU's -45.6%.
#
# WHAT THE CHANGE DOES, unchanged from v3: the connector indexes block_ids with
# GLOBAL token positions -- `starts // block_size` in
# ChunkedTokenDatabase.prepare_values -- while the worker built one database per
# group from the raw spec.block_size. Under DCP one physical block of an
# ATTENTION group spans block_size * dcp global tokens; a Mamba group's state is
# replicated per rank and still spans block_size. resolve_kv_cache_block_sizes
# states exactly that rule. PCP is still refused: different sharding, not
# mirrored by this scaling, never measured.
#
# WHAT THIS SCRIPT CANNOT CHECK. Whether Mooncake writes a byte. Pipeline
# 61788267 ran a full Mooncake ladder in which every one of 122,449 batch_put
# calls failed with TRANSFER_FAIL and the CPU hit rate was 0.0% at all eleven
# concurrencies, and it finished green -- it measured a server with no offload
# and called it Mooncake. Read the worker log for batch_put failures and CPU hit
# rate before believing any number from these arms.

set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-merged-v4.sh"

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
