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
# ROOT CAUSE, measured (pipeline 65163609). Each PP stage owns a different set of
# layers, so each sees a different KV cache group layout -- different count AND
# different order:
#
#   PP0  groups=5  [Mamba, Mamba, Mamba, Mamba, MLA]   -> mamba ids [0, 1, 2, 3]
#   PP1  groups=3  [MLA,   Mamba, Mamba]               -> mamba ids [1, 2]
#
# All 16 workers were checked: no worker ever sees two layouts. The caller's cached
# list is therefore NOT stale -- it reads its own config correctly. What is wrong is
# the context, which reported [0, 1, 2] on both stages, matching neither.
#
# A previous version of this patch forced the context's list everywhere. That is worse
# than the bug: on PP1 the real layout is [MLA, Mamba, Mamba], so [0, 1, 2] treats the
# MLA group as mamba -- which is exactly the "all mamba block tables must share
# stride(0), got {2064, 683}" mixture, self-inflicted.
#
# THE FIX. The list freshly derived from the config in hand is authoritative. So derive
# it, and if the cached context was built against a different one, rebuild the context
# instead of bending the indices to fit it. Then num_groups, the block-table slice and
# the ordering of compute_aligned_state_indices all come from the same layout.
#
# It logs all three lists at the moment they are compared -- fresh, context, caller --
# so a run says which of them moved rather than only that they disagreed.
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

if "[mambagroups2]" in src:
    print("[mambagroups2] already applied")
    sys.exit(0)

IMPORT = """from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
"""
if src.count(IMPORT) != 1:
    sys.exit("[mambagroups2] FATAL: expected one mamba_utils import block, found %d" % src.count(IMPORT))
src = src.replace(IMPORT, """from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    get_mamba_groups,
""", 1)

SLICE = """            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                self.model.get_mamba_state_copy_func(),
                [block_tables[gid] for gid in mamba_group_ids],
            )
"""
if src.count(SLICE) != 1:
    sys.exit(
        "[mambagroups2] FATAL: expected one initialize_from_forward_context call, found "
        "%d -- the align context was restructured and this patch needs re-deriving"
        % src.count(SLICE)
    )
src = src.replace(SLICE, """            # [mambagroups2] The list derived from the config in hand is authoritative:
            # each PP stage has its own layout and the cached context may have been
            # built against another one. Rebuild rather than reindex.
            fresh_ids = list(get_mamba_groups(kv_cache_config)[0])
            ctx_ids = list(ctx.mamba_group_ids)
            if ctx_ids != fresh_ids:
                logger.warning(
                    "[mambagroups2] stale ctx: fresh=%s ctx=%s caller=%s | cfg groups=%d "
                    "specs=%s -- rebuilding the context",
                    fresh_ids,
                    ctx_ids,
                    list(mamba_group_ids),
                    len(kv_cache_config.kv_cache_groups),
                    [type(g.kv_cache_spec).__name__ for g in kv_cache_config.kv_cache_groups],
                )
                self._mamba_ctx = MambaSpecDecodeGPUContext.create(
                    max_num_reqs=self.max_num_reqs,
                    kv_cache_config=kv_cache_config,
                    num_state_types=len(self.model.get_mamba_state_copy_func()),
                    device=self.device,
                    make_buffer=lambda n, dtype: CpuGpuBuffer(
                        n, dtype=dtype, device=self.device
                    ),
                )
                ctx = self._mamba_ctx
                ctx_ids = list(ctx.mamba_group_ids)
                if ctx_ids != fresh_ids:
                    raise AssertionError(
                        "[mambagroups2] a freshly built context still disagrees: "
                        f"fresh={fresh_ids} ctx={ctx_ids}"
                    )
            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                self.model.get_mamba_state_copy_func(),
                [block_tables[gid] for gid in fresh_ids],
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
        "[mambagroups2] FATAL: %d align-mode builder blocks, expected 0 or 1" % _have_block
    )
if _have_block == 0:
    print("[mambagroups2] no #52388 builder block in this image; slice fix only")
src = src.replace(BLOCK, """        if self._align_mode:
            # [mambagroups2] Decide whether any builder wants aligned indices, using a
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
                for group_idx, group_id in enumerate(
                    get_mamba_groups(kv_cache_config)[0]
                ):
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
    sys.exit("[mambagroups2] FATAL: expected one caller cache assignment, found %d" % src.count(CACHE))
src = src.replace(CACHE, """            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
            # [mambagroups2] which config this cache was built from
            self._mg_cached_cfg_id = hex(id(kv_cache_config))
            logger.warning(
                "[mambagroups2] caller cache built | cfg id=%s groups=%d specs=%s "
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
        sys.exit("[mambagroups2] FATAL: no VllmConfig import to anchor the logger to")
    src = src.replace(IMP, "from vllm.config import VllmConfig\nfrom vllm.logger import init_logger\n", 1)
    # Anchor above the decorator, not above the class: inserting between @dataclass
    # and its class detaches the decorator and the file stops parsing.
    CLS = "@dataclass\nclass MambaHybridAttnMetadata("
    if src.count(CLS) != 1:
        sys.exit("[mambagroups2] FATAL: no decorated MambaHybridAttnMetadata to anchor to")
    src = src.replace(CLS, "logger = init_logger(__name__)\n\n\n" + CLS, 1)

compile(src, target, "exec")
open(target, "w").write(src)
print("[mambagroups2] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(root, "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
src = open(path).read()

if "[block_tables[gid] for gid in mamba_group_ids]" in src:
    sys.exit("[mambagroups2] FATAL: a slice by the caller's list is still there")
if "[block_tables[gid] for gid in fresh_ids]" not in src:
    sys.exit("[mambagroups2] FATAL: the fresh-list slice is missing")
if "get_mamba_groups," not in src:
    sys.exit("[mambagroups2] FATAL: get_mamba_groups was not imported")
if "rebuilding the context" not in src:
    sys.exit("[mambagroups2] FATAL: the stale-context rebuild is missing")
# Only meaningful on an image that has #52388's builder block. If it was never there,
# neither marker is expected -- but the old caller-indexed list must not be either.
if "aligned_index_builders" in src:
    sys.exit("[mambagroups2] FATAL: the old caller-indexed builder list survived")
if "needs_aligned" in src and (
    "get_mamba_groups(kv_cache_config)[0]\n                ):" not in src
):
    sys.exit("[mambagroups2] FATAL: the builder loop is not driven by the fresh list")
# Two: the inline comment at the slice and the log format. The substantive checks are
# the four above and below; this only catches a partial write.
if src.count("[mambagroups2]") < 2:
    sys.exit("[mambagroups2] FATAL: markers missing after write (%d)" % src.count("[mambagroups2]"))

import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

# The field being read has to exist, or this trades one crash for another.
for f in ("mamba_group_ids", "num_groups"):
    if f not in getattr(MambaSpecDecodeGPUContext, "__dataclass_fields__", {}):
        sys.exit("[mambagroups2] FATAL: MambaSpecDecodeGPUContext has no %s field" % f)
if not hasattr(mh, "logger"):
    sys.exit("[mambagroups2] FATAL: module logger missing")

print("[mambagroups2] verified: ctx fields exist, slice uses them, logger present")
PY

echo "=== mambagroups: done ==="
