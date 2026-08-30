#!/usr/bin/env bash
# The mamba group-id cache is keyed on nothing, and the config it answers for changes.
# =============================================================================
# This replaces mambagroups2, which worked but touched three call sites and rewrote a
# loop. Measurement showed the whole defect is one unsound cache, so this is the smaller
# and more defensible form -- and the one worth sending upstream.
#
# WHAT BREAKS. On the 2026-08-28 nightly (6f7df92a), K3 with mamba-cache-mode align and
# DCP8 dies in cudagraph capture, before serving:
#
#   mamba_hybrid.py  _ensure_align_ctx
#   mamba_utils.py   assert len(block_tables) == self.num_groups
#   AssertionError: expected 3 block tables, got 4
#   AssertionError: all mamba block tables must share stride(0), got {2064, 683}
#
# 683 is the attention block-table width, so a non-mamba group is being read as mamba.
#
# THE CAUSE, measured. `_get_mamba_group_info` computes the mamba group ids once and
# caches them on `self`, with no key:
#
#   if self._mamba_spec is None:
#       ... derive from kv_cache_config.kv_cache_groups ...
#
# Everything else derives the same list fresh, through `get_mamba_groups(kv_cache_config)`.
# Within a single worker process we logged the two answers:
#
#   cache built  -> [0, 1, 2, 3]   from a config with 5 groups [M, M, M, M, MLA]
#   ctx created  -> [0, 1, 2]      from the config in hand at that later moment
#
# So the config the cache answers for is not the config in hand. Upstream #52388 added a
# second call site in `prepare_attn`, which runs during capture -- earlier than the
# original one in `preprocess_state`, which dummy batches skip -- and that is the first
# time the cache is populated at a different moment from everything that reads around it.
#
# THE FIX. Key the cache on the config it was built from. Same list for repeated calls
# with the same config, so no per-step cost; recomputed when the config changes, which
# is the case that was silently wrong. Nothing outside this function reads
# `_mamba_group_ids` or `_mamba_spec` -- checked -- so the change is contained.
#
# WHAT IT IS NOT. Not a revert of #52388, which is a 6.6-7.6x kernel improvement. Not a
# claim about which layout is correct: each PP stage has its own, and this simply makes
# every reader see the one belonging to the config it was handed.
#
# Accuracy is not assumed. The mambagroups2 form of this fix was gated on GSM8K at c48 --
# 0.9462 no-spec and 0.9492 ns=4 with block rejection, against 0.950 on the pinned image --
# and this arm re-runs that gate rather than inheriting it.
# =============================================================================
set -euo pipefail

echo "=== mambacache: key the mamba group-id cache on the config it came from ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
)
src = open(target).read()

if "[mambacache]" in src:
    print("[mambacache] already applied")
    sys.exit(0)
if "[mambagroups" in src:
    sys.exit(
        "[mambacache] FATAL: a mambagroups patch is already applied; the two are "
        "alternatives and must not be chained"
    )

ANCHOR = """        if self._mamba_spec is None:
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[mambacache] FATAL: expected one unconditional mamba group-id cache guard, "
        "found %d" % src.count(ANCHOR)
    )

src = src.replace(
    ANCHOR,
    """        # [mambacache] Key the cache on the config it was derived from. Each PP
        # stage has its own KV cache layout, and this function is called at more than
        # one moment; caching the first answer forever hands a later caller ids that
        # belong to a config it is not looking at.
        if self._mamba_spec is None or getattr(self, "_mamba_cfg", None) is not kv_cache_config:
""",
    1,
)

SET = """            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
"""
if src.count(SET) != 1:
    sys.exit("[mambacache] FATAL: expected one cache assignment, found %d" % src.count(SET))
src = src.replace(
    SET,
    """            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
            self._mamba_cfg = kv_cache_config
""",
    1,
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[mambacache] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(root, "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
src = open(path).read()

if "if self._mamba_spec is None:\n" in src:
    sys.exit("[mambacache] FATAL: the unkeyed cache guard is still there")
if '_mamba_cfg", None) is not kv_cache_config' not in src:
    sys.exit("[mambacache] FATAL: the config key is missing from the guard")
if "self._mamba_cfg = kv_cache_config" not in src:
    sys.exit("[mambacache] FATAL: the config key is never stored")

# Contained by construction: nothing outside this function may read the cached fields,
# or keying the cache would change behaviour somewhere this patch has not looked.
import re

body = re.search(
    r"def _get_mamba_group_info\(.*?\n(?=    def |\nclass )", src, re.S
).group(0)
outside = src.replace(body, "")
for f in ("_mamba_group_ids", "_mamba_spec"):
    hits = [
        ln for ln in outside.splitlines()
        if f in ln and "self._mamba_group_ids: list[int] = []" not in ln
        and "self._mamba_spec: MambaSpec | None = None" not in ln
    ]
    if hits:
        sys.exit("[mambacache] FATAL: %s is read outside the function: %r" % (f, hits[:2]))

import vllm.v1.worker.gpu.model_states.mamba_hybrid  # noqa: F401

print("[mambacache] verified: guard keyed on the config, key stored, fields contained")
PY

echo "=== mambacache: done ==="
