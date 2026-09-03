#!/usr/bin/env bash
# Stop #50611 from promoting the CP interleave for a store-only connector.
# =============================================================================
# WHAT BREAKS. vllm#50611 ([Nixl][PD] DCP support for MLA models, 7f4793eaa3)
# added VllmConfig.adjust_dcp_kv_cache_interleave_size(), which sets
#
#     cp_kv_cache_interleave_size = local_block_size
#
# whenever decode_context_parallel_size > 1 and *any* kv_connector is configured.
# It asks only whether a connector exists, not whether we are doing P/D. Our
# aggregated arms set MooncakeStoreConnector -- an external prefix-cache tier,
# not a transfer path -- so the promotion fires on plain AGG, and on K3 it lands
# on the mamba block size:
#
#     cp_kv_cache_interleave_size is automatically adjusted from 1 to
#     block_size 1536 for block-level alignment.
#
# MooncakeStoreScheduler cannot account for blocks under that interleave and the
# engine dies during the first AgentX warmup request:
#
#     scheduler.py:424 _apply_current_save_block_ids
#     AssertionError: Missing current block table for store request chatcmpl-...
#
# Measured on pipeline 66111736: the no-spec arm reached READY and died here, and
# the promotion is not gated on speculation, so every arm is affected. The 08/28
# image has no adjust_dcp_kv_cache_interleave_size at all, which is why the same
# recipe has been fine there -- this is a regression between the two images, not
# a property of our config.
#
# THE FIX is the predicate vllm#54457 introduces from the other side:
# requires_dcp_block_aligned_interleave on KVConnectorBase_V1, False for
# connectors that do not move blocks across workers. That PR is not merged and,
# per our own review of it, does not cover MooncakeStoreConnector anyway. So we
# short-circuit the promotion for the store connector here, by name, and leave
# every other connector -- including NIXL, which #50611 was written for -- alone.
#
# This is deliberately narrower than "disable the promotion": a P/D arm on NIXL
# still gets block-aligned interleave, because there it is load-bearing.
# =============================================================================
set -euo pipefail

echo "=== mooncake-no-interleave-promo: keep interleave 1 for the store connector ==="

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
target = os.path.join(root, "vllm/config/vllm.py")
src = open(target).read()

if "adjust_dcp_kv_cache_interleave_size" not in src:
    print("[mooncake-interleave] this image predates #50611; nothing to gate")
    sys.exit(0)

MARK = "_k3_store_connector_skips_interleave_promotion"
if MARK in src:
    print("[mooncake-interleave] already gated in this image")
    sys.exit(0)

ANCHOR = """        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector is not None
            and self.parallel_config.cp_kv_cache_interleave_size != local_block_size
        ):
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[mooncake-interleave] FATAL: expected one promotion guard in "
        "adjust_dcp_kv_cache_interleave_size, found %d" % src.count(ANCHOR)
    )

GATED = """        # _k3_store_connector_skips_interleave_promotion: MooncakeStoreConnector
        # is an external prefix-cache tier, not a cross-worker transfer path, and
        # its scheduler cannot account for blocks under a promoted interleave --
        # it raises "Missing current block table for store request" on the first
        # request. #50611 promotes for any connector; #54457 would add the
        # predicate that stops it, but is unmerged and misses this connector.
        _store_only = {"MooncakeStoreConnector"}
        _connector = self.kv_transfer_config.kv_connector if self.kv_transfer_config else None
        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector is not None
            and _connector not in _store_only
            and self.parallel_config.cp_kv_cache_interleave_size != local_block_size
        ):
"""
src = src.replace(ANCHOR, GATED, 1)
compile(src, target, "exec")
open(target, "w").write(src)
print("[mooncake-interleave] applied: " + target)
PY

# Verify the gate exists and that the promotion is still reachable for other
# connectors -- a patch that disabled it outright would break NIXL P/D arms.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/config/vllm.py")).read()
if "adjust_dcp_kv_cache_interleave_size" not in src:
    print("[mooncake-interleave] verified: nothing to do on this image")
    sys.exit(0)
if "_k3_store_connector_skips_interleave_promotion" not in src:
    sys.exit("[mooncake-interleave] FATAL: the gate is not in the file after writing it")
if "cp_kv_cache_interleave_size = local_block_size" not in src:
    sys.exit("[mooncake-interleave] FATAL: the promotion itself was removed, not gated")
print("[mooncake-interleave] verified: gated for the store connector, live for others")
PY

echo "=== mooncake-no-interleave-promo: done ==="
