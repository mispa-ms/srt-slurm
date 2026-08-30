#!/usr/bin/env bash
# Bound the GPU blocks #51358 pins for Mooncake boundary-state saves.
# =============================================================================
# WHAT WE MEASURED. #51358 ("Save exact Mamba boundary states", merged
# 2026-08-29T02:40Z) costs 9-13% on this workload, twice, independently:
#
#   08/28 baseline          9,409.6 tok/s/chip   gpu hit 86.2%   catch 55.7%   miss 6.58%
#   08/28 + the patch       8,556.4   (-9.0%)    gpu hit 86.4%   catch 26.8%   miss 9.98%
#   08/30, merged in image  8,201.8  (-12.8%)    gpu hit 71.6%   catch 64.0%   miss 10.22%
#
# WHY. The scheduler pins every block a store job touches so it cannot be recycled
# before all ranks have written it:
#
#   self._pinned_saves[store_job_id] = (block_ids, self._num_workers)
#   pool.touch([pool.blocks[block_id] for block_id in block_ids])
#
# and BlockPool.touch removes the block from free_block_queue and raises ref_cnt, so a
# pinned block is no longer an eviction candidate -- it is out of the prefix cache for
# as long as the pin lasts, which is until all 16 of our ranks report the save done.
#
# Two things make that expensive here rather than incidental:
#
#   1. THE PIN IS NOT LIMITED TO THE BOUNDARY BLOCKS. After collecting the exact
#      boundary-state blocks it adds every allocated block of the request, across all
#      non-Mamba groups -- the whole attention/MLA block table. The comment gives the
#      reason (a lagging rank may resume below the current range), but on a trace whose
#      ISL is p50 89k / p90 289k that is a very large pin per request.
#
#   2. THERE IS NO CAP, although the PR describes one. Its own design notes say "The
#      configurable in-flight cap bounds pinned GPU blocks if acknowledgements stop;
#      saves beyond the cap remain best-effort and are dropped." No such cap is in the
#      merged code: _pinned_saves is only ever declared, assigned, read, decremented,
#      deleted and tested for emptiness -- its size is never checked -- and envs.py has
#      no Mooncake pin knob.
#
# WHAT THIS DOES. Implements the cap the PR specifies. Before pinning, if this job would
# take the total pinned block count past the limit, the save is skipped and nothing is
# pinned. That is the stated best-effort behaviour: a dropped save costs a later cache
# miss, whereas an unbounded pin costs the prefix cache continuously.
#
# It does NOT relax the correctness the PR bought. Every save that is pinned is pinned
# exactly as before, so no block can be recycled underneath a write in flight. The
# dropped ones are not written at all, rather than written from a block that moved.
#
# CAP. K3_PINCAP_BLOCKS, default 8192 blocks. Chosen as a starting point, not a tuned
# value: it is meant to be small against the pool and large enough that steady-state
# saves are unaffected. The log line reports every drop, so an arm that never logs was
# never near the cap and an arm that logs constantly needs a bigger one.
#
# HOW TO READ THE ARM. gpu_cache_hit_rate is the signal. On 08/30 it fell 86.2% -> 71.6%
# with no change in pool size; if the pin is the cause it should come back toward 86%.
# If it does not, the pin is not the mechanism and this whole line of reasoning is wrong.
# =============================================================================
set -euo pipefail

echo "=== k3-pincap: bound the blocks pinned for Mooncake boundary-state saves ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py",
)
src = open(target).read()

if "[k3-pincap]" in src:
    print("[k3-pincap] already applied")
    sys.exit(0)

ANCHOR = """            self._pinned_saves[store_job_id] = (block_ids, self._num_workers)
            pool.touch([pool.blocks[block_id] for block_id in block_ids])
"""
if src.count(ANCHOR) == 0:
    print("[k3-pincap] no boundary-state pinning in this image; nothing to bound")
    sys.exit(0)
if src.count(ANCHOR) != 1:
    sys.exit("[k3-pincap] FATAL: expected one pin site, found %d" % src.count(ANCHOR))

REPLACEMENT = """            # [k3-pincap] Bound the pin, which #51358 specifies and does not
            # implement: "The configurable in-flight cap bounds pinned GPU blocks
            # if acknowledgements stop; saves beyond the cap remain best-effort and
            # are dropped." Without it every allocated block of every in-flight
            # save leaves free_block_queue, and on a long-context trace that is a
            # large and continuous subtraction from the prefix cache.
            _k3_cap = int(os.environ.get("K3_PINCAP_BLOCKS", "8192"))
            _k3_pinned_now = sum(len(b) for b, _ in self._pinned_saves.values())
            if _k3_pinned_now + len(block_ids) > _k3_cap:
                logger.warning_once(
                    "[k3-pincap] dropping a boundary-state save: %d blocks pinned "
                    "+ %d requested exceeds K3_PINCAP_BLOCKS=%d",
                    _k3_pinned_now,
                    len(block_ids),
                    _k3_cap,
                )
                continue
            self._pinned_saves[store_job_id] = (block_ids, self._num_workers)
            pool.touch([pool.blocks[block_id] for block_id in block_ids])
"""
src = src.replace(ANCHOR, REPLACEMENT, 1)

# The replacement uses os.environ, and this module imports nothing bare -- every line
# is `from vllm... import ...` -- so anchor on the first of those rather than on
# "\nimport ", which is not in the file at all.
if "\nimport os\n" not in src:
    FIRST = "from vllm.config import VllmConfig\n"
    if src.count(FIRST) != 1:
        sys.exit("[k3-pincap] FATAL: no unique import anchor to insert `import os` at")
    src = src.replace(FIRST, "import os\n\n" + FIRST, 1)

compile(src, target, "exec")
open(target, "w").write(src)
print("[k3-pincap] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(
    root, "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py"
)
src = open(path).read()

if "[k3-pincap]" not in src:
    print("[k3-pincap] verified: image has no pin to bound")
    raise SystemExit(0)

for need in ("K3_PINCAP_BLOCKS", "_k3_pinned_now", "logger.warning_once"):
    if need not in src:
        sys.exit(f"[k3-pincap] FATAL: {need} missing after the write")

# The cap has to sit BEFORE the pin, or it bounds nothing.
cap_at = src.index("_k3_pinned_now + len(block_ids) > _k3_cap")
pin_at = src.index("pool.touch([pool.blocks[block_id] for block_id in block_ids])")
if cap_at > pin_at:
    sys.exit("[k3-pincap] FATAL: the cap check lands after the pin")

import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler  # noqa: F401

print(
    "[k3-pincap] verified: cap precedes the pin, env knob present, module imports "
    f"(K3_PINCAP_BLOCKS={os.environ.get('K3_PINCAP_BLOCKS', '8192 (default)')})"
)
PY

echo "=== k3-pincap: done ==="
