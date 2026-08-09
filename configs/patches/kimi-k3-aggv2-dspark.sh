#!/bin/bash
# Kimi-K3 AGG v2 + DSpark speculative decoding under DCP.
#
# WHY THIS IS A RUNTIME PATCH AND NOT A REBUILD. v2's discipline is that vLLM
# source changes are compiled into the image, and it is the right default -- it
# is what stops a patch from silently never reaching the run, which is how the
# Mamba block-table defect survived for weeks on the old track. The exception
# here is narrow and deliberate: these six files are pure Python, they touch no
# CUDA and no build input, so rebuilding would cost 1-2 hours of kernel
# compilation to produce a bit-identical binary. The check block below is the
# price of the exception -- it re-verifies the DSpark change AND every v2 marker
# before any GPU time is spent.
#
# WHAT IS PATCHED, and nothing else. From
# mispa-ms/vllm@misunp/k3-dcp-dspark-v2 -- our six files, the flatten switch
# (default off), and vllm#50532:
#
#   config/speculative.py            drop the DSpark x DCP refusal
#   kimi_k3/nvidia/mla.py            drop the RoPE x DCP assert (prefill-CP
#                                    refusal stays)
#   cp_utils.py                      the rank-local KV slot formula as one
#                                    triton device function -- additive, nothing
#                                    existing changes
#   dflash/speculator.py             the draft indexed the block table with
#                                    positions // block_size and used the raw
#                                    offset as its slot. Under DCP one block
#                                    covers block_size * cp_size global tokens,
#                                    so the draft wrote its KV elsewhere. Both
#                                    sites, plus its DCP-local seq lens.
#   dflash/cudagraph.py              the same seq lens on the capture path --
#                                    the draft has two metadata paths and
#                                    neither was DCP-aware
#   spec_decode/speculator.py        forward the field
#
# DELIBERATELY NOT CARRIED. The per-query causal-bound flatten: the kernel
# already computes that bound and measured at c8/dcp8 the flatten cost ITL p90
# 8.4% while LOWERING acceptance (1.587 against 1.622). The coordinator hoist:
# it existed because wzhao-d87cdf5ce4 made verify_and_split_kv_cache_groups read
# enable_partial_hash_hits, d87 is gone from v2, and that method reads the flag
# zero times on this base. block_table.py's refactor and the various
# instrumentation: diagnostics for closed defects.
#
# CONFIG SIDE, not here. Under MRV2 the uniform-decode graphs come from
# round_up(size, q_len) // q_len, and compilation.py's
# adjust_cudagraph_sizes_for_spec_decode is gated on `not use_v2_model_runner`,
# which is false here -- so a q_len=1 capture list leaves holes in the
# request-count grid and a speculative batch landing in one falls to PIECEWISE
# at a flat ~59 ms. Every DSpark config on this branch lists
# q_len*1 .. q_len*max_num_seqs. That is checked over the yml before submit and
# again post-hoc from the worker log, not here -- see the note at the end.

set -euo pipefail

bash /configs/patches/kimi-k3-aggv2.sh

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

FIX=/configs/patches/k3-dspark-on-aggv2.patch
echo "=== k3-dspark-on-aggv2 ==="
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

# --- the four behaviours this patch exists for ---
assert 'MLA DSpark does not currently support decode context' not in read('config/speculative.py')
mla = read('models/kimi_k3/nvidia/mla.py')
assert 'does not support RoPE with decode' not in mla, 'RoPE x DCP assert still present'
assert 'does not support prefill context' in mla, (
    'the prefill-CP refusal was removed; only the decode-CP one should be gone'
)

df = read('v1/worker/gpu/spec_decode/dflash/speculator.py')
# The half that is easy to lose in a merge: a CP-aware slot with a non-CP block
# index reads the wrong row of the block table and is silently wrong.
assert df.count('// (block_size * CP_SIZE)') == 2, (
    f'expected both dflash slot sites to index with block_size * CP_SIZE, '
    f'got {df.count(chr(47)+chr(47)+chr(32)+chr(40)+chr(98))}'
)
assert df.count('cp_local_slot(') == 2, 'both dflash slot sites must use the helper'
assert 'def cp_local_slot' in read('v1/worker/gpu/cp_utils.py')

# Both draft metadata paths, not one. Only the serve path was ever DCP-aware.
assert 'prepare_dcp_local_seq_lens(' in df, 'draft serve path'
assert 'prepare_dcp_local_seq_lens(' in read('v1/worker/gpu/spec_decode/dflash/cudagraph.py'), (
    'draft cudagraph capture path'
)
assert 'dcp_local_seq_lens' in read('v1/worker/gpu/spec_decode/speculator.py')

# --- the flatten switch, and the invariant it must not break ---
ts = read('v1/attention/backends/mla/tokenspeed_mla.py')
assert 'supports_non_causal_multi_token_decode: ClassVar[bool] = True' in ts, (
    'vllm#50911 missing -- without it the draft is forced off TOKENSPEED_MLA and '
    'the DCP reorder threshold collapses to 1'
)
flat = [l for l in ts.splitlines() if l.startswith('_DCP_FLATTEN = os.environ.get')]
assert len(flat) == 1 and 'VLLM_TS_MLA_DCP_FLATTEN' in flat[0], (
    'the per-query causal-bound flatten switch is missing'
)
assert flat[0].split(',')[1].strip().strip(chr(34) + chr(39)).startswith('0'), (
    'the per-query causal-bound flatten switch is missing, or no longer defaults off'
)

# q must be reshaped to 4D on EVERY path into the kernel. The reshape lives in
# an if/elif/else whose head is the flatten branch, so the earlier check that it
# was a top-level statement is now wrong -- the real invariant is that every
# branch of the chain assigns q from .view/.unsqueeze. Getting this wrong hands
# the kernel a 3D query: an unpack error 15 minutes into startup, spec arms only.
fn = next(n for n in ast.walk(ast.parse(ts))
          if isinstance(n, ast.FunctionDef) and n.name == 'forward_mqa')

def reshapes_q(body):
    for st in body:
        for n in ast.walk(st):
            if (isinstance(n, ast.Assign)
                    and any(getattr(t, 'id', None) == 'q' for t in n.targets)
                    and isinstance(n.value, ast.Call)
                    and getattr(n.value.func, 'attr', '') in ('view', 'unsqueeze')):
                return True
    return False

chain = None
for st in fn.body:
    if not isinstance(st, ast.If):
        continue
    c, node = [], st
    while True:
        c.append(node)
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
        else:
            break
    if any('num_decode_tokens % num_decodes' in ast.unparse(x.test) for x in c):
        chain = c
        break
assert chain is not None, 'forward_mqa has no if-chain containing the uneven-length test'
bad = [ast.unparse(x.test)[:60] for x in chain if not reshapes_q(x.body)]
assert not bad, f'these branches reach the kernel without reshaping q: {bad}'
assert reshapes_q(chain[-1].orelse), 'the final else does not reshape q'

# vllm#50532 is carried: MRV2 classified a batch as uniform decode from its
# SHAPE alone, so a prompt chunk could be dispatched FULL and replay the
# spec-verify decode graph. Measured at DCP=1 on GSM8K: 0.936 without it against
# 0.948 no-spec, and 0.948 with it -- full recovery, FULL cudagraphs kept.
assert 'get_uniform_decode_token_count' in read('v1/worker/gpu/model_runner.py'), (
    'vllm#50532 missing -- the scheduler path still classifies uniform decode by '
    'shape alone, which costs 1.2 GSM8K points at DCP=1 and 2.7 at DCP=8'
)
assert 'def is_uniform_query_len' in read('v1/worker/utils.py')

# --- v2 itself must be untouched by all of the above ---
for rel, marker, who in [
    ('v1/worker/gpu/model_runner.py', 'kv_shard_count = 1 if isinstance(spec, MambaSpec)', 'ours/v2'),
    ('models/kimi_k3/nvidia/kda_metadata.py', 'def _check_block_table_width', 'ours/v2'),
    ('v1/simple_kv_offload/manager.py', 'def _group_block_size', 'ours/v2'),
    ('v1/core/sched/scheduler.py', 'req_hybrid_block_ids = {', 'ours/v2'),
    ('v1/core/kv_cache_coordinator.py',
     'dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size', '#50493'),
    ('v1/attention/ops/dcp_utils.py', 'class MLADCPManager', '#50484'),
]:
    assert marker in read(rel), f'{who} lost from {rel}: {marker}'

print('DSpark-on-v2 verified in', root)
"

# The capture-size grid is a property of the yml, not of the image, and nothing
# at setup time can see the CLI args the server will get. It is validated where
# it can be: by tools/check_capture_grid.py over the config files before submit,
# and post-hoc from the worker log, where
#   Capturing CUDA graphs (FULL): N/N
# must have N == max_num_seqs. A runtime check here could only read env vars the
# config would have to duplicate, and would silently pass when they were unset.
