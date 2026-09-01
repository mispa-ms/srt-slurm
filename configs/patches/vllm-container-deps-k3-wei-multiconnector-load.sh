#!/usr/bin/env bash
# Wei's bb9afd157, re-derived onto #53324: MultiConnector load ownership.
# =============================================================================
# THE BUG. Under MultiConnector, several children each report a hit and the
# scheduler picks one. MooncakeStoreScheduler.update_state_after_alloc then does
# two wrong things when it is not the one chosen:
#
#   1. it records local_block_ids = () -- so the blocks that WERE allocated are
#      lost, and this connector still owns the save for them;
#   2. it sets can_load = False on a load spec it keeps, rather than dropping the
#      spec, leaving an unchosen load in its table.
#
# Wei's comment is the clearest statement of the fix: "Keep the real allocation
# for saves, but discard this connector's unchosen load." The two halves are a
# pair; taking either alone leaves the other failure.
#
# ONLY REACHABLE UNDER MultiConnector, which is the disagg recipe: NixlConnector
# for the P-to-D handshake plus MooncakeStoreConnector for external DRAM KV. The
# aggregated arms name MooncakeStoreConnector directly, so nothing there can
# select a different child and this patch is inert for them.
#
# THIS ONE IS RE-DERIVED, NOT CHERRY-PICKED -- read this before trusting it.
# bb9afd157 is written against the pre-#53324 form of the first hunk:
#
#     local_block_ids: tuple[list[int], ...] = ()
#     if num_external_tokens > 0:
#         local_block_ids = blocks.get_block_ids()
#
# #53324 rewrote that same line to route through
# kv_cache_config.select_transfer_block_ids(), so Wei's hunk does not apply and
# the guard has to be removed from the newer form instead. The second hunk is
# his, byte for byte. Everything else on this stack is a cherry-pick; this is the
# one place a human decided what the commit meant, which is why it is in its own
# file and its own arm rather than folded into the compact-I/O patch.
# =============================================================================
set -euo pipefail

echo "=== wei-multiconnector-load: keep the allocation, drop the unchosen load ==="

PATCH=/configs/patches/k3-wei-multiconnector-load.patch
SITE=$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin)))")
SCHED="$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py"

if grep -q "self.load_specs.pop(request.request_id)" "$SCHED"; then
  echo "[wei-multiconnector-load] already present"
else
  cd "$SITE"
  patch -p1 --forward --no-backup-if-mismatch --fuzz=0 < "$PATCH"
  echo "[wei-multiconnector-load] applied"
fi

python3 - <<'PY'
import ast
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(
    root, "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py")
src = open(path).read()
fn = next((n for n in ast.walk(ast.parse(src))
           if isinstance(n, ast.FunctionDef) and n.name == "update_state_after_alloc"),
          None)
if fn is None:
    sys.exit("[wei-multiconnector-load] FATAL: update_state_after_alloc is gone")
body = ast.unparse(fn)

# Both halves, checked on the parsed function rather than by grepping the file,
# so a hunk that landed in some other method cannot pass.
if "if num_external_tokens > 0:" in body:
    sys.exit("[wei-multiconnector-load] FATAL: the allocation is still guarded; "
             "a request whose load another child won would record no blocks and "
             "this connector still owns their save")
if "select_transfer_block_ids" not in body:
    sys.exit("[wei-multiconnector-load] FATAL: select_transfer_block_ids is no "
             "longer called; #53324's group-aware block selection was dropped")
if "self.load_specs.pop(request.request_id)" not in body:
    sys.exit("[wei-multiconnector-load] FATAL: the unchosen load spec is not popped")
if "can_load = False" in body:
    sys.exit("[wei-multiconnector-load] FATAL: the old can_load flag survives "
             "alongside the pop; only one of the two can be right")

import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler  # noqa: F401

print("[wei-multiconnector-load] verified: allocation unguarded, group selection "
      "kept, unchosen load popped, old flag gone")
PY

echo "=== wei-multiconnector-load: done ==="
