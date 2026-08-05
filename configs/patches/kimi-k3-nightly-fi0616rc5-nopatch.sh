#!/bin/bash
# kimi-k3-nightly-fi0616rc5.sh with the hybrid-KV recompute patch REVERTED, so
# the image runs stock nightly scheduler code.
#
# WHY THIS EXISTS. The base rc5 script has always called
# patch_kimi_k3_mooncake_hma_recompute.py behind a `|| echo NOTE`. Until the file
# was added to this branch that call was a no-op, which is why the -cur- ladder --
# including the 3,626 tok/s/GPU c16 reference -- genuinely ran zero patches. Now
# the file is here, so the base script applies it and "no patches" is no longer
# reachable by picking the base script.
#
# This script reverts that one edit after the base script runs, using the
# patcher's own constants so the two can never drift. It exists purely so an arm
# can price all three patches at once against a patched arm.
#
# EXPECT IT TO CRASH IF THE PATH IS HIT. Without the revert-target in place, a KV
# load failure on a hybrid model raises
# `ValueError: too many values to unpack (expected 1)` at
# scheduler.py:_update_requests_with_invalid_blocks and takes EngineCore with it
# (pipeline 61123650). A crash here is a result: it means the patch is not an
# optimisation but a prerequisite.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5.sh

echo "=== reverting the hybrid-KV recompute patch (stock nightly scheduler) ==="
python3 - <<'PY'
import importlib.util
import pathlib

import vllm

spec = importlib.util.spec_from_file_location(
    "hma", "/configs/patches/patch_kimi_k3_mooncake_hma_recompute.py")
hma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hma)

p = pathlib.Path(vllm.__file__).parent / "v1/core/sched/scheduler.py"
src = p.read_text()

if hma.NEW_HYBRID_RECOVERY in src:
    src = src.replace(hma.NEW_HYBRID_RECOVERY, hma.OLD_SINGLE_GROUP_RECOVERY, 1)
    compile(src, str(p), "exec")
    p.write_text(src)
    print("reverted hybrid-KV recompute patch in", p)
else:
    print("hybrid-KV recompute patch was not present; nothing to revert")

src = p.read_text()
assert "req_hybrid_block_ids = {" not in src, "revert failed: patch marker still present"
assert "(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)" in src, \
    "revert failed: the stock single-group unpack is not back"
print("stock nightly scheduler verified in", p)
PY
