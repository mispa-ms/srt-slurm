#!/bin/bash
# The new-container DCP stack plus the Mamba block-table fix.
#
# THE DEFECT. vllm/v1/worker/gpu/model_runner.py sizes every KV cache group's
# block table with
#
#     max_num_blocks = cdiv(max_model_len, spec.block_size * self.dcp_size)
#
# applying the DCP divisor to Mamba groups as well. Mamba state is replicated
# across DCP ranks and never sharded, and MambaSpec.max_num_blocks_per_req says
# so in its own comment and returns cdiv(max_len, block_size) instead. The two
# disagree by exactly dcp_size. Measured on this stack at DCP=8:
#
#     spec.max_num_blocks_per_req = 683      what MambaSpec says it needs
#     ACTUAL_bt_width             =  86      what the runner allocated
#     cdiv(max_len, block_size)     = 683
#     cdiv(max_len, block_size*dcp) =  86
#
# kda_metadata.py's _get_aligned_state_indices_kernel then reads the block table
# at column (seq_lens - 1) // block_size -- up to 682 -- with a mask that bounds
# only the row. Past 86 * 1536 = 132,096 tokens it reads off the end of the row,
# and the arbitrary int32 it picks up is used as a state block id. When that id
# lands outside recurrent_state (2,454 blocks here) the run dies on
#
#     IndexKernel.cu:111 "index out of bounds"
#
# at kda.py:732, or at kda.py:682 -> causal_conv1d_fn, whichever consumer of the
# index tensor runs first. When the stray id happens to be in range there is no
# assert at all -- just wrong recurrent state, which is the failure mode of
# upstream #51039 on this same model.
#
# WHY IT LOOKED LIKE A DCP-SCALE BUG. The threshold is bt_width * block_size =
# max_model_len / dcp_size:
#
#     dcp 1   683 cols   1,049,088 tokens   = max_model_len, unreachable
#     dcp 2   342         525,312
#     dcp 4   171         262,656
#     dcp 8    86         132,096
#
# Our ISL is mean 129,544 / p90 239,783, so DCP=8 crosses it constantly, DCP=4
# rarely, DCP=2 and DCP=1 never. That is the observed failure order exactly:
# dcp1 375 requests clean, dcp2 364 clean, dcp4 one arm of eighteen, dcp8 six of
# twenty-five.
#
# THE FIX. Two parts.
#
#   1. model_runner.py takes kv_shard_count = 1 for MambaSpec, so the Mamba
#      block table is sized the way MambaSpec.max_num_blocks_per_req already
#      specifies. The Triton kernel is left alone: dividing by block_size was
#      always right, the table was wrong.
#
#   2. kda_metadata.py checks the received width against
#      max_num_blocks_per_req once, at the first build, and raises. A clamp in
#      the kernel would convert the crash into silent state corruption, which is
#      strictly worse; this fails at startup and names the cause.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5-hma-dcp-newctr.sh

echo "=== Mamba block-table DCP fix (ours) ==="
PATCH_FILE=/configs/patches/k3-mamba-blocktable-dcp-fix.patch
SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")

if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "block-table fix already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH_FILE"
fi

python3 -c "
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
for rel, marker in [
    ('v1/worker/gpu/model_runner.py', 'kv_shard_count = 1 if isinstance(spec, MambaSpec) else self.dcp_size'),
    ('models/kimi_k3/nvidia/kda_metadata.py', 'def _check_block_table_width'),
]:
    src = (root / rel).read_text()
    assert marker in src, f'block-table fix missing in {rel}: {marker}'
print('Mamba block-table DCP fix verified in', root)
"
