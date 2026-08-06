#!/bin/bash
# kimi-k3-nightly-fi0616rc5-hma-pmu-dcp.sh minus wzhao18/vllm@d87cdf5ce4.
#
# One delta against that script, so an arm on it and an arm on this one differ by
# exactly Wei Zhao's prefix-match-unit patch and nothing else.
#
# WHY. Wei asked directly, in the B300 agg thread: "The prefix-match-unit patch I
# had might still be immature. I wonder if you run without the patch (and simply
# using PMU), if you would see similar perf." This is that run.
#
# WHAT COMES OUT. Two outcomes and both are answers:
#
#   it runs      the patch is not load-bearing for this configuration and the
#                comparison is a straight perf delta.
#   it asserts   kv_cache_manager.truncate_computed_blocks fails on
#                `num_blocks <= len(group_blocks)`, which is the assert his patch
#                replaces with null-block padding. Then the patch is a
#                prerequisite here, not an optimisation, and that is worth
#                telling him.
#
# The second is a real possibility: this arm meets every condition his commit
# names -- Kimi-K3 is hybrid, mamba_cache_mode is align, a KV connector is
# active, and prefix-match-unit 128 is finer than the 1536 block. Against that,
# our own k3-dcp-offload-hybrid-fix.patch floors the CPU-tier hit to the
# scheduler block, which may remove the lag his patch pads around. Nobody has
# run the combination, so the run decides.
#
# NOT THE SAME AS DROPPING prefix-match-unit. The config keeps pmu 128. Only the
# patch is gone.

set -euo pipefail

DCP_BASE_SCRIPT=/configs/patches/kimi-k3-nightly-fi0616rc5-hma.sh \
    bash /configs/patches/kimi-k3-nightly-fi0616rc5-dcp.sh

python3 -c "
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent

# The point of this script is that the prefix-match-unit patch is absent. Fail
# closed if something put it back, or the arm silently stops being the control.
mgr = (root / 'v1/core/kv_cache_manager.py').read_text()
assert 'spec.mamba_cache_mode == \"align\"' not in mgr, (
    'wzhao18/vllm@d87cdf5ce4 is present; this arm is supposed to run without it'
)
print('confirmed: running without the prefix-match-unit patch')
"
