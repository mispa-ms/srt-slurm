#!/bin/bash
# The new-container DCP stack plus KDA state-index instrumentation.
#
# DIAGNOSTIC STACK. Identical to kimi-k3-nightly-fi0616rc5-hma-dcp-newctr.sh
# with one patch added on top; nothing else moves, so a run on this script is
# comparable with the -c4-...-newctr- arm it is chasing.
#
# WHAT IT ADDS AND WHY
#
# CUDA_LAUNCH_BLOCKING localised the crash to
#
#   kimi_k3/nvidia/kda.py:732
#     recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
#
# and, with prefix caching off, to kda.py:682 -> causal_conv1d_fn, which takes
# the same tensor as cache_indices. So the index tensor itself carries an
# out-of-range block id. What is NOT known is which value, for which row, and
# against which bound -- and it cannot be read back after the fact, because the
# device-side assert kills the CUDA context.
#
# k3-kda-state-index-debug.patch validates the tensor on the host BEFORE
# anything indexes the state cache with it, and logs, on the first offence:
#
#   n_state_blocks        recurrent_state.shape[0], the real bound
#   conv_state_blocks     the other consumer's bound
#   bad_pos / bad_val     which entries are out of range and what they hold
#   seq_lens_at_bad       the seq_lens those rows were derived from
#   slot_at_bad           (seq_lens - 1) // cache_block, the slot the Triton
#                         kernel read the block table at
#   bt_width              the block table's column count
#   raw_block_table_row   the offending row's first and last columns
#   prefills / decodes / spec / token counts
#
# Those numbers decide which axis is off. Adding a clamp to the kernel without
# them would turn the crash into silent state corruption, which is worse.
#
# COST. One GPU->host sync per step, not per layer: all 69 KDA layers share the
# one index tensor the metadata builder produced, so the check runs only for the
# first KDA layer it sees. A healthy sample line is emitted every 500 steps so
# the normal slot/index range is visible even if nothing trips.
#
# OFF BY DEFAULT. Both halves are gated on VLLM_KDA_STATE_INDEX_DEBUG=1; the
# config must set it. This script checks for that at the end.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5-hma-dcp-newctr.sh

echo "=== KDA state-index instrumentation (ours, diagnostic only) ==="
PATCH_FILE=/configs/patches/k3-kda-state-index-debug.patch
SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")

if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "KDA debug patch already applied"
else
    patch -p1 --forward -d "$SITE" < "$PATCH_FILE"
fi

python3 -c "
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
for rel, marker in [
    ('models/kimi_k3/nvidia/kda.py', 'def _kda_debug_check_state_indices'),
    ('models/kimi_k3/nvidia/kda.py', 'KDA-IDX BAD step=%d'),
    ('models/kimi_k3/nvidia/kda_metadata.py', 'md.dbg_seq_lens = m.seq_lens'),
]:
    src = (root / rel).read_text()
    assert marker in src, f'KDA debug patch missing in {rel}: {marker}'
print('KDA state-index instrumentation verified in', root)
"

if [ "${VLLM_KDA_STATE_INDEX_DEBUG-}" != "1" ]; then
    echo "FATAL - VLLM_KDA_STATE_INDEX_DEBUG is '${VLLM_KDA_STATE_INDEX_DEBUG-<unset>}', must be 1."
    echo "This stack exists only to run that check; without it the patch is inert"
    echo "and the run is an expensive duplicate of the plain -newctr- arm."
    exit 1
fi
echo "KDA state-index debug enabled"
