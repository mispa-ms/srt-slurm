#!/usr/bin/env bash
# Promote the DCP interleave for P/D transfer connectors only, not for every connector.
# =============================================================================
# WHAT BREAKS. On the 2026-08-30 nightly (1dc464d426), K3 with DCP8 + DSpark + a Mooncake
# store dies at engine init, before serving:
#
#   assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
#   AssertionError: MTP with cp_kv_cache_interleave_size > 1 is not supported
#                   in TokenspeedMLAImpl
#
# FlashInfer MLA refuses the same thing, so no attention backend avoids it:
#
#   if dcp_size > 1 and interleave_size != 1:
#       raise ValueError("FlashInfer MLA native DCP requires
#                         cp_kv_cache_interleave_size=1; got ...")
#
# WHERE THE VALUE COMES FROM. Not from us -- cp_kv_cache_interleave_size defaults to 1 and
# we never set it. config/vllm.py raises it in adjust_dcp_kv_cache_interleave_size:
#
#   if (self.kv_transfer_config is not None
#       and self.kv_transfer_config.kv_connector is not None
#       and self.parallel_config.cp_kv_cache_interleave_size != local_block_size):
#           self.parallel_config.cp_kv_cache_interleave_size = local_block_size
#
# The guard fires on the mere presence of a connector. Its own log line says "When using
# PD disaggregation with DCP", and the code it came from -- #50611, "[Nixl][PD] DCP
# support for MLA models", merged 2026-08-29T16:03Z -- is about exactly that: a NIXL
# producer that shards MLA KV across DCP ranks, where the consumer needs block-level
# alignment to read a remote rank once. The condition never checks whether any of that is
# happening.
#
# WHY IT IS WRONG FOR US. We run a single aggregated instance. MooncakeStoreConnector is
# a prefix-cache offload tier, not a P/D transfer: no KV crosses instances, so there is
# no remote rank to align for. And the connector already handles DCP on its own -- that
# is what #53324, "Support MooncakeStore with hybrid DCP prefix caching", is.
#
# THE EVIDENCE THAT INTERLEAVE 1 IS CORRECT HERE. The promotion does not exist in
# 6f7df92a (08/28) or 6d4562c59b (08/29), so on both images the value stays at its
# default 1, and on those images this exact configuration runs:
#
#   Mooncake external catch 55.7%   (c48 ns=4, 3,988 requests, err 0.25%)
#   GSM8K 0.9492                    (c48 ns=4, block rejection)
#
# So DCP8 + Mooncake + interleave 1 is not merely tolerated, it is measured correct.
#
# WHAT THIS CHANGES. One condition. The promotion now also requires the connector to be
# a P/D transfer instance by role AND not to be a store-style offload connector. There is
# no existing predicate that separates the two -- is_kv_transfer_instance is true for us
# as well, since we declare kv_role kv_both -- which is the root of the bug and the thing
# worth raising upstream. Until upstream has one, the connector name is the honest
# discriminator, and it is checked explicitly rather than inverted, so an unknown
# connector keeps the current behaviour.
#
# WHAT IT DOES NOT DO. It does not touch NIXL, LMCache, or any other connector, and it
# does not change the interleave for anything that is doing P/D. If the guard was
# protecting something we have not understood, the accuracy arm is what says so -- this
# is submitted with a GSM8K twin for that reason, because external_cache_hit_rate stays
# high whether the bytes are right or wrong.
# =============================================================================
set -euo pipefail

echo "=== k3-interleave: promote the DCP interleave for P/D connectors only ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/config/vllm.py",
)
src = open(target).read()

if "[k3-interleave]" in src:
    print("[k3-interleave] already applied")
    sys.exit(0)

ANCHOR = """        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector is not None
            and self.parallel_config.cp_kv_cache_interleave_size != local_block_size
        ):
"""
if src.count(ANCHOR) == 0:
    # Nothing to narrow: this image predates #50611, so the promotion is not here and
    # the interleave stays at its default. That is the 08/28 and 08/29 case.
    print("[k3-interleave] no connector interleave promotion in this image; nothing to do")
    sys.exit(0)
if src.count(ANCHOR) != 1:
    sys.exit(
        "[k3-interleave] FATAL: expected one connector interleave promotion, found %d"
        % src.count(ANCHOR)
    )

REPLACEMENT = """        # [k3-interleave] Only promote for a connector that actually moves KV between
        # instances. #50611 added this for NIXL P/D, where a DCP-sharded producer has to
        # be readable block-at-a-time by a remote consumer, but the condition fired on
        # the presence of any connector. A store-style offload tier has no remote rank
        # to align for, and MooncakeStoreConnector handles DCP itself since #53324.
        # Raising it anyway makes both MLA backends refuse the config outright.
        _k3_conn = (
            self.kv_transfer_config.kv_connector
            if self.kv_transfer_config is not None
            else None
        )
        _k3_offload_only = _k3_conn in ("MooncakeStoreConnector",)
        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector is not None
            and not _k3_offload_only
            and self.parallel_config.cp_kv_cache_interleave_size != local_block_size
        ):
"""
src = src.replace(ANCHOR, REPLACEMENT, 1)
compile(src, target, "exec")
open(target, "w").write(src)
print("[k3-interleave] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(root, "vllm/config/vllm.py")
src = open(path).read()

if "[k3-interleave]" not in src:
    print("[k3-interleave] verified: image has no promotion to narrow")
    raise SystemExit(0)

if "_k3_offload_only" not in src:
    sys.exit("[k3-interleave] FATAL: the narrowing variable is missing after the write")

# The original condition must be gone: if both survive, one of them still promotes.
BARE = """            and self.kv_transfer_config.kv_connector is not None
            and self.parallel_config.cp_kv_cache_interleave_size != local_block_size
        ):
"""
if BARE in src:
    sys.exit("[k3-interleave] FATAL: an un-narrowed promotion is still present")

import vllm.config.vllm  # noqa: F401

print("[k3-interleave] verified: promotion narrowed, module imports")
PY

echo "=== k3-interleave: done ==="
