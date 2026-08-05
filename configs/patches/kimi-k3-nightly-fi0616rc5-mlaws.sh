#!/bin/bash
# kimi-k3-nightly-fi0616rc5.sh plus the MLA chunked-prefill workspace patch.
#
# One delta against the script it wraps, so an arm using this and an arm using
# the base script differ only by wzhao18/vllm@2331dddd94 + @d4f1b6438c.
#
# What the patch buys: the workspace stops scaling with max_num_seqs. Unpatched,
# Kimi-K3's block_size of 1536 makes the "1 page per request" floor override the
# 64k cap for any max_num_seqs >= 43, at ~6.7 KB/token/GPU on TP8 -- 420 MB at
# the cap against 10.5 GB at max_num_seqs=1024. Patched, max_num_seqs becomes a
# free knob, which is what Wei Zhao's GB300 config leans on when it leaves
# max-num-seqs unset at the vLLM default of 1024.
#
# The patch is not upstream and carries no upstream review; its correctness
# rests on the claim that the gather kernels accept unaligned chunk starts. A
# wrong claim shows up as wrong output, not as a crash, so compare generated
# text against a base-script run before reading any timing from this.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5.sh

echo "=== MLA chunked-prefill workspace patch ==="
python3 - <<'PY'
import vllm
from vllm.model_executor.layers.attention.mla_attention import MLACommonBackend
print("vllm", vllm.__version__, "at", vllm.__file__)
PY
python3 /configs/patches/patch_mla_chunked_prefill_workspace.py
python3 -c "
import vllm, pathlib
p = pathlib.Path(vllm.__file__).parent / 'model_executor/layers/attention/mla_attention.py'
src = p.read_text()
assert 'tokens_per_request = cache_config.block_size' in src, 'workspace floor not patched'
assert 'align_chunk_to_block=self.dcp_world_size > 1' in src, 'align flag not patched'
print('MLA workspace patch verified in', p)
"
