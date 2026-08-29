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
# THE FIX. Drive the whole block from one list -- the context's. It is what num_groups
# counts and what compute_aligned_state_indices orders its output by, so slicing and
# indexing with it cannot disagree with itself. It needs no claim about which of the two
# lists is "right", only that one of them is the context's own.
#
# A first version fixed just the block_tables slice. That got past the first assertion
# and straight into the second, because the same stale list also drives the builder
# loop: group_idx comes from enumerating the caller's list while all_group_indices is
# ordered by the context's, so builders are assigned the wrong group's indices and one
# is never assigned at all -- "Aligned Mamba state indices must be precomputed".
# Measured: caller=[0, 1, 2, 3] and [1, 2, 3] against ctx=[0, 1, 2].
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

SLICE = """            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                self.model.get_mamba_state_copy_func(),
                [block_tables[gid] for gid in mamba_group_ids],
            )
"""
if src.count(SLICE) != 1:
    sys.exit(
        "[mambagroups] FATAL: expected one initialize_from_forward_context call, found "
        "%d -- the align context was restructured and this patch needs re-deriving"
        % src.count(SLICE)
    )
src = src.replace(SLICE, """            # [mambagroups] Slice with the context's own group ids -- the list its
            # num_groups counts. And record which config each list came from: the two
            # functions are semantically identical, so if they disagree it is because
            # they ran against different configs at different moments, and that is the
            # root cause rather than the mismatch itself.
            _mg_specs = [
                type(g.kv_cache_spec).__name__ for g in kv_cache_config.kv_cache_groups
            ]
            logger.warning(
                "[mambagroups] ctx created here | cfg id=%s groups=%d specs=%s | "
                "cached_at_id=%s",
                hex(id(kv_cache_config)),
                len(kv_cache_config.kv_cache_groups),
                _mg_specs,
                getattr(self, "_mg_cached_cfg_id", None),
            )
            ctx_group_ids = list(ctx.mamba_group_ids)
            if list(mamba_group_ids) != ctx_group_ids:
                logger.warning(
                    "[mambagroups] group ids disagree: caller=%s ctx=%s "
                    "(num_groups=%d) -- using the ctx list",
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
""", 1)

# The builder loop drives group_idx from the caller's list while all_group_indices is
# ordered by the context's. Same divergence, second symptom.
BLOCK = """        if self._align_mode:
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            aligned_index_builders = []
            for group_idx, group_id in enumerate(mamba_group_ids):
                for group in attn_groups[group_id]:
                    builder = group.get_metadata_builder(0)
                    if hasattr(builder, "mamba_aligned_state_indices"):
                        aligned_index_builders.append((group_idx, builder))
            if aligned_index_builders:
                ctx = self._ensure_align_ctx(
                    kv_cache_config, mamba_group_ids, block_tables
                )
                all_group_indices = ctx.compute_aligned_state_indices(
                    input_batch.seq_lens, num_reqs
                )
                for group_idx, builder in aligned_index_builders:
                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]
"""
# The builder block is #52388's addition and exists only on nightlies that carry it.
# On an image without it there is nothing to correct here and the slice fix above is the
# whole patch -- skip rather than fail, so this script is safe on either nightly.
_have_block = src.count(BLOCK)
if _have_block > 1:
    sys.exit(
        "[mambagroups] FATAL: %d align-mode builder blocks, expected 0 or 1" % _have_block
    )
if _have_block == 0:
    print("[mambagroups] no #52388 builder block in this image; slice fix only")
src = src.replace(BLOCK, """        if self._align_mode:
            # [mambagroups] Decide whether any builder wants aligned indices, using a
            # freshly derived list; then assign them driven by ctx.mamba_group_ids,
            # because all_group_indices is ordered by that list and by nothing else.
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            needs_aligned = any(
                hasattr(group.get_metadata_builder(0), "mamba_aligned_state_indices")
                for group_id in mamba_group_ids
                for group in attn_groups[group_id]
            )
            if needs_aligned:
                ctx = self._ensure_align_ctx(
                    kv_cache_config, mamba_group_ids, block_tables
                )
                all_group_indices = ctx.compute_aligned_state_indices(
                    input_batch.seq_lens, num_reqs
                )
                for group_idx, group_id in enumerate(ctx.mamba_group_ids):
                    for group in attn_groups[group_id]:
                        builder = group.get_metadata_builder(0)
                        if hasattr(builder, "mamba_aligned_state_indices"):
                            builder.mamba_aligned_state_indices = all_group_indices[
                                group_idx
                            ]
""", 1)

# Record which config populated the caller's cache, so the two moments can be compared.
CACHE = """            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
"""
if src.count(CACHE) != 1:
    sys.exit("[mambagroups] FATAL: expected one caller cache assignment, found %d" % src.count(CACHE))
src = src.replace(CACHE, """            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
            # [mambagroups] which config this cache was built from
            self._mg_cached_cfg_id = hex(id(kv_cache_config))
            logger.warning(
                "[mambagroups] caller cache built | cfg id=%s groups=%d specs=%s "
                "-> ids=%s",
                self._mg_cached_cfg_id,
                len(kv_cache_config.kv_cache_groups),
                [type(g.kv_cache_spec).__name__ for g in kv_cache_config.kv_cache_groups],
                group_ids,
            )
""", 1)

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
# Only meaningful on an image that has #52388's builder block. If it was never there,
# neither marker is expected -- but the old caller-indexed list must not be either.
if "aligned_index_builders" in src:
    sys.exit("[mambagroups] FATAL: the old caller-indexed builder list survived")
if "needs_aligned" in src and (
    "for group_idx, group_id in enumerate(ctx.mamba_group_ids):" not in src
):
    sys.exit("[mambagroups] FATAL: the builder loop is not driven by the ctx list")
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
