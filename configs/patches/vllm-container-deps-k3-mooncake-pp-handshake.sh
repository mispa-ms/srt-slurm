#!/usr/bin/env bash
# MooncakeStoreConnector accepts PP-sharded handshake metadata.
# =============================================================================
# WHAT THIS UNBLOCKS. Pipeline-parallel disaggregated serving with a Mooncake tier.
# Without it engine-core init dies on every rank past the first stage:
#
#   set_xfer_handshake_metadata_pp_aware ... rejects pp_rank > 0
#
# The base class refuses PP-sharded handshake metadata to protect connectors that
# read peer transfer-agent metadata. MooncakeStoreConnector does not read it --
# producers and consumers meet in MooncakeDistributedStore, and its inherited plain
# setter returns None -- so the refusal kills startup over a value it discards. The
# fix is a no-op override.
#
# Found and first shipped by the GB300 PP-disagg session inside its own engine patch
# (k3-engine-0819.patch, derived against image commit 5a4c8d9924). That patch stack
# does not apply to the images this workstream runs, so the thirty lines are lifted
# out and re-expressed here.
#
# WHY AN INSERTION AND NOT A DIFF. Three patches on this workstream broke today
# because a diff was carried to an image whose file had drifted. This edits by
# content: it finds the class, checks the method is absent, and inserts before
# `def shutdown`. The line numbers differ between the two images we target
# (connector.py:172 on 728d3ad, :160 on a9a17e70) and this does not care.
# =============================================================================
set -euo pipefail

echo "=== mooncake-pp-handshake: let the store connector accept PP shards ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py",
)
src = open(target).read()

if "set_xfer_handshake_metadata_pp_aware" in src:
    print("[mooncake-pp-handshake] already present: " + target)
    sys.exit(0)

if "class MooncakeStoreConnector" not in src:
    sys.exit(
        "[mooncake-pp-handshake] FATAL: MooncakeStoreConnector not found in %s; the "
        "connector moved and this needs re-deriving" % target
    )

# 1. the type the override is annotated with
IMPORT_ANCHOR = "    KVConnectorMetadata,\n"
if "KVConnectorHandshakeMetadata" not in src:
    if src.count(IMPORT_ANCHOR) != 1:
        sys.exit(
            "[mooncake-pp-handshake] FATAL: expected one KVConnectorMetadata import "
            "line, found %d" % src.count(IMPORT_ANCHOR)
        )
    src = src.replace(
        IMPORT_ANCHOR, "    KVConnectorHandshakeMetadata,\n" + IMPORT_ANCHOR, 1
    )

# 2. the override itself, in front of shutdown, at that method's own indent
SHUTDOWN = "    def shutdown(self):"
if src.count(SHUTDOWN) != 1:
    sys.exit(
        "[mooncake-pp-handshake] FATAL: expected one `def shutdown` in the connector, "
        "found %d" % src.count(SHUTDOWN)
    )

METHOD = '''    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """Discard peer handshake metadata, PP shards included.

        Producers and consumers meet in MooncakeDistributedStore, not through each
        other's transfer agents, so this connector never reads handshake metadata --
        the base class's plain setter for it is already a no-op here. The base
        PP-aware setter still rejects ``pp_rank > 0`` to protect connectors that do
        read it, which stops PP-disaggregated serving from starting over a value this
        one throws away.
        """

'''

src = src.replace(SHUTDOWN, METHOD + SHUTDOWN, 1)
compile(src, target, "exec")
open(target, "w").write(src)
print("[mooncake-pp-handshake] applied: " + target)
PY

echo "=== mooncake-pp-handshake: done ==="
