#!/usr/bin/env bash
# One source of truth for mamba_group_ids, instead of two that can disagree.
# =============================================================================
# WHAT BREAKS. On the 2026-08-28 nightly (6f7df92a), K3 with mamba-cache-mode align and
# DCP8 dies during cudagraph capture, before serving:
#
#   capture_model -> prepare_inputs_to_capture -> prepare_attn
#   mamba_hybrid.py:283  _ensure_align_ctx
#   mamba_utils.py:1001  assert len(block_tables) == self.num_groups
#   AssertionError: expected 3 block tables, got 4
#
# and on the other stage, the follow-on:
#
#   AssertionError: all mamba block tables must share stride(0), got {2064, 683}
#
# 683 is the attention block-table width, so one of the four tables handed to the mamba
# path is not a mamba table at all.
#
# WHY. There are two independently computed copies of the same list:
#
#   ctx.mamba_group_ids    get_mamba_groups(kv_cache_config), recomputed inside
#                          MambaSpecDecodeGPUContext.create(); num_groups is its length
#   self._mamba_group_ids  mamba_hybrid._get_mamba_group_info(), computed once and
#                          cached on first call
#
# _ensure_align_ctx slices block_tables with the *cached* one and the assertion compares
# against the *recomputed* one. Upstream #52388 ("Optimize k3 mamba metadata
# preparation") added a second call site in prepare_attn, which runs during capture --
# earlier than the original one in preprocess_state, which is skipped on dummy batches.
# That is the first time the two are established at different moments, and on this
# nightly they come out different.
#
# THE FIX. Slice with the list the context was actually built from, which is the same
# list its num_groups counts. The two cannot disagree by construction, and it needs no
# claim about which list is "right" -- only that one of them is the context's own.
#
# AND IT SAYS WHEN IT MATTERED. If the two lists are identical this patch is a no-op,
# and a run that then succeeds succeeded for some other reason. So it logs the
# difference the first time it sees one. No log line means the lists agreed and this was
# not the fix -- a distinction that has been worth a day on this workstream before.
#
# Not a revert: #52388 is a 6.6-7.6x kernel improvement and stays.
# =============================================================================
set -euo pipefail

echo "=== mambagroups: slice block tables with the context's own group ids ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
)
src = open(target).read()

if "[mambagroups]" in src:
    print("[mambagroups] already applied")
    sys.exit(0)

ANCHOR = """            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                self.model.get_mamba_state_copy_func(),
                [block_tables[gid] for gid in mamba_group_ids],
            )
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[mambagroups] FATAL: expected one initialize_from_forward_context call, found "
        "%d -- the align context was restructured and this patch needs re-deriving"
        % src.count(ANCHOR)
    )

ADDITION = """            # [mambagroups] Slice with the context's own group ids. The caller's
            # copy is cached from its first call, the context's is recomputed in
            # create(), and num_groups counts the context's -- so slicing with the
            # caller's is what makes them disagree.
            ctx_group_ids = list(ctx.mamba_group_ids)
            if list(mamba_group_ids) != ctx_group_ids:
                logger.warning(
                    "[mambagroups] group ids disagree: caller=%s ctx=%s "
                    "(num_groups=%d) -- slicing with the ctx list",
                    list(mamba_group_ids),
                    ctx_group_ids,
                    ctx.num_groups,
                )
            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                self.model.get_mamba_state_copy_func(),
                [block_tables[gid] for gid in ctx_group_ids],
            )
"""

src = src.replace(ANCHOR, ADDITION, 1)

# This module has no logger of its own -- it is a worker-state module and every other
# one here logs through its caller. The warning has to be able to fire, so bring one.
# The nullguard patch hit exactly this and it is worth checking for every time.
if "logger = init_logger(__name__)" not in src:
    IMP = "from vllm.config import VllmConfig\n"
    if src.count(IMP) != 1:
        sys.exit("[mambagroups] FATAL: no VllmConfig import to anchor the logger to")
    src = src.replace(IMP, "from vllm.config import VllmConfig\nfrom vllm.logger import init_logger\n", 1)
    # Anchor above the decorator, not above the class: inserting between @dataclass
    # and its class detaches the decorator and the file stops parsing.
    CLS = "@dataclass\nclass MambaHybridAttnMetadata("
    if src.count(CLS) != 1:
        sys.exit("[mambagroups] FATAL: no decorated MambaHybridAttnMetadata to anchor to")
    src = src.replace(CLS, "logger = init_logger(__name__)\n\n\n" + CLS, 1)

compile(src, target, "exec")
open(target, "w").write(src)
print("[mambagroups] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(root, "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
src = open(path).read()

if "[block_tables[gid] for gid in mamba_group_ids]" in src:
    sys.exit("[mambagroups] FATAL: a slice by the caller's list is still there")
if "[block_tables[gid] for gid in ctx_group_ids]" not in src:
    sys.exit("[mambagroups] FATAL: the ctx-list slice is missing")
# Two: the inline comment at the slice and the log format. The substantive checks are
# the four above and below; this only catches a partial write.
if src.count("[mambagroups]") < 2:
    sys.exit("[mambagroups] FATAL: markers missing after write (%d)" % src.count("[mambagroups]"))

import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

# The field being read has to exist, or this trades one crash for another.
for f in ("mamba_group_ids", "num_groups"):
    if f not in getattr(MambaSpecDecodeGPUContext, "__dataclass_fields__", {}):
        sys.exit("[mambagroups] FATAL: MambaSpecDecodeGPUContext has no %s field" % f)
if not hasattr(mh, "logger"):
    sys.exit("[mambagroups] FATAL: module logger missing")

print("[mambagroups] verified: ctx fields exist, slice uses them, logger present")
PY

echo "=== mambagroups: done ==="
