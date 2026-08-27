#!/usr/bin/env bash
# Seed the align-mode mamba state block with the mamba block size, not the scheduler's.
# =============================================================================
# THIS IS UPSTREAM PR #53798, which is open and unmerged, applied to the nightly we run.
# Its own regression test states the failure exactly:
#
#   "the seed divided by the SCHEDULER block size (16) instead of the MAMBA block size
#    (880 after page unification on hybrid models). A request resumed from a retained
#    checkpoint at 107,360 tokens was seeded to column 6,709 instead of 121 -- the
#    first align precopy then dereferenced a garbage block-table column: a neighbouring
#    request's state at moderate lengths (silent corruption), unmapped memory at ~100k+
#    (CUDA illegal memory access)."
#
# Every clause of that matches what we measured on PP2:
#
#   align mode           our logs carry "Padding mamba page size ... align"
#   column far too large our [ckptcol] probe: 7 of 8 columns past a table of width 683
#   garbage column       state_idx feeds conv_state's address arithmetic directly
#   silent corruption    why guarding one consumer did not stop the crash
#   illegal access ~100k AgentX prompts run to 470k tokens at p95
#
# WHY SIX FIXES MISSED IT. Every one of them guarded a *consumer* of the bad column --
# the checkpoint kernel's grid, its gather, the metadata builder. This is the
# *producer*. The same wrong column is also handed to the align precopy, which nothing
# on this branch touched, so the fault simply moved there.
#
#   shared metadata builder     no relation
#   cp_gather_cache bounds      different kernel
#   grid length (3 tensors)     lengths agree; the values were wrong
#   grid length (4 tensors)     same
#   checkpoint column mask      masked one consumer, not the source
#   token_idx negative          already masked by valid_conv; caught before submitting
#
# THE CHANGE. add_request seeds the running state block as
#
#   (num_computed_tokens - 1) // self.cache_config.block_size
#
# with cache_config.block_size being the scheduler's. On a hybrid model the mamba group
# has its own, larger block size, already resolved into self._mamba_spec. Use it, and
# fall back to the old divisor only before the spec exists -- which is only reachable
# for a fresh request, where num_computed_tokens is 0 and the result is -1 either way.
#
# A fresh request is unaffected. A resumed one lands on its own row instead of far
# outside it.
# =============================================================================
set -euo pipefail

echo "=== mambaseed: seed the align state block with the mamba block size ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
)
src = open(target).read()

if "[mambaseed]" in src:
    print("[mambaseed] already applied: " + target)
    sys.exit(0)

ANCHOR = """            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[mambaseed] FATAL: expected one align-mode seed, found %d; either the code "
        "moved or PR #53798 already landed" % src.count(ANCHOR)
    )

ADDITION = '''            # [mambaseed] upstream PR #53798. cache_config.block_size is the
            # scheduler's; on a hybrid model the mamba group has its own, larger
            # block size after page unification. Dividing by the scheduler's points
            # the precopy source outside this request's block-table row -- a
            # neighbour's state at moderate lengths, unmapped memory past ~100k.
            _ms_spec = getattr(self, "_mamba_spec", None)
            _ms_block = (
                _ms_spec.block_size
                if _ms_spec is not None
                else self.cache_config.block_size
            )
            _ms_seed = (new_req_data.num_computed_tokens - 1) // _ms_block
            if (
                new_req_data.num_computed_tokens > 0
                and _ms_block != self.cache_config.block_size
                and _MAMBASEED["n"] < 20
            ):
                _MAMBASEED["n"] += 1
                logger.warning(
                    "[mambaseed] #%d computed=%d | mamba_block=%d -> col %d | "
                    "scheduler_block=%d would have given %d",
                    _MAMBASEED["n"],
                    new_req_data.num_computed_tokens,
                    _ms_block,
                    _ms_seed,
                    self.cache_config.block_size,
                    (new_req_data.num_computed_tokens - 1)
                    // self.cache_config.block_size,
                )
            self._mamba_state_idx_gpu[req_index].fill_(_ms_seed)
'''

src = src.replace(ANCHOR, ADDITION, 1)

# The module has no logger of its own -- checked, not assumed -- so add one along with
# the report cap. Without it the warning above is a NameError.
if "init_logger" not in src:
    IMPORT_ANCHOR = "from vllm.config import VllmConfig\n"
    if src.count(IMPORT_ANCHOR) != 1:
        sys.exit(
            "[mambaseed] FATAL: expected one VllmConfig import to anchor the logger "
            "to, found %d" % src.count(IMPORT_ANCHOR)
        )
    src = src.replace(
        IMPORT_ANCHOR, IMPORT_ANCHOR + "from vllm.logger import init_logger\n", 1
    )

DEF_ANCHOR = "\n@dataclass\n"
if DEF_ANCHOR not in src:
    sys.exit("[mambaseed] FATAL: no dataclass to place the module state before")
src = src.replace(
    DEF_ANCHOR,
    '\nlogger = init_logger(__name__)\n\n# [mambaseed] report cap\n_MAMBASEED = {"n": 0}\n'
    + DEF_ANCHOR,
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[mambaseed] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")).read()
if src.count("[mambaseed]") < 3:
    sys.exit("[mambaseed] FATAL: markers missing after write")
if "// self.cache_config.block_size\n            )" in src:
    sys.exit("[mambaseed] FATAL: the old seed expression is still the one used")

import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh

for name in ("logger", "_MAMBASEED"):
    if name not in dir(mh):
        sys.exit("[mambaseed] FATAL: %s missing from the module" % name)
print("[mambaseed] verified: module imports, seed uses the mamba block size")
PY

echo "=== mambaseed: done ==="
