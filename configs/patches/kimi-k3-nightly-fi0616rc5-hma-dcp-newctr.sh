#!/bin/bash
# The weiport DCP stack on the 2026-08-07 nightly: hma base, no d87.
#
# Same shape as kimi-k3-nightly-fi0616rc5-hma-dcp.sh -- the hybrid-KV recompute
# patch, then DCP, and no prefix-match-unit patch -- but pointed at the rebased
# DCP script. See kimi-k3-nightly-fi0616rc5-dcp-newctr.sh for what moved.
#
# The hma anchor was checked against c810e5ee before this was written:
# OLD_SINGLE_GROUP_RECOVERY is still present in v1/core/sched/scheduler.py, so
# patch_kimi_k3_mooncake_hma_recompute.py applies unchanged.
#
# d87cdf5ce4 stays out for the same reason as on the old container: these arms
# are the control half of the with/without-d87 pair. It has since merged
# upstream, but not into this container -- checked, and the assert below is what
# keeps that true if a later nightly picks it up.

set -euo pipefail

DCP_BASE_SCRIPT=/configs/patches/kimi-k3-nightly-fi0616rc5-hma.sh \
    bash /configs/patches/kimi-k3-nightly-fi0616rc5-dcp-newctr.sh

python3 -c "
import pathlib
import vllm
root = pathlib.Path(vllm.__file__).parent
mgr = (root / 'v1/core/kv_cache_manager.py').read_text()
assert 'spec.mamba_cache_mode == \"align\"' not in mgr, (
    'wzhao18/vllm@d87cdf5ce4 is present; this arm is supposed to run without it'
)
print('confirmed: running without the prefix-match-unit patch')
"
