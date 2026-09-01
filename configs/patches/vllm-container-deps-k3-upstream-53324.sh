#!/usr/bin/env bash
# Replace our Mooncake DCP carry with upstream vllm#53324, which is better.
# =============================================================================
# WHAT THIS SWAPS, AND WHY IT IS A SWAP RATHER THAN AN ADDITION.
#
# A Mooncake key is a whole block, so the store holds an object only at each
# group's block boundaries. Core's fine-grained lookup deliberately extends a
# hit INTO the first non-full block, and with EAGLE/DSpark then subtracts one
# hash unit, landing mid-block by construction. Off DCP that gap is empty.
# Scaling the attention block by dcp opens a block_size/hash_block_size-wide
# interior, and every hit landing there names a key nobody wrote: measured on
# B300 c8 DCP=8, 2,757,664 OBJECT_NOT_FOUND (-704), the run stuck in warmup for
# four hours because kv_load_failure_policy=recompute retried forever.
#
# We fixed that by revalidating the reconciled boundary and stepping the hit
# back until its key existed. #53324 fixes the same defect the other way round:
# it keeps the longer hit and resolves, per group, the hash boundary whose key
# was actually used to store that tail block.
#
# OURS IS NOT MERELY REDUNDANT WITH #53324 -- IT IS WORSE. Ours throws away a
# hit upstream can now load. That sentence has been sitting in the header of
# test_mooncake_dcp_keyset.py since we wrote the carry, and has never been
# measured; this script is what makes measuring it possible.
#
# #53324 merged 2026-08-29 02:10 UTC, one day after the 6f7df92a8e nightly this
# stack pins, so it is not in the image and has to be applied. It is applied
# here as the upstream commit verbatim, not as a re-derivation.
#
# WHAT THIS FILE ASSUMES ABOUT THE CHAIN. kimi-k3-nightly-v8.sh must have
# applied k3-ours6-v8-nomc.patch -- our v8 carry with its three Mooncake files
# removed -- and not k3-ours6-v8-nightly.patch. The two cannot both be applied:
# they rewrite the same functions. The check below refuses rather than fuzzes.
#
# ON THE PCP REFUSAL. Our carry narrowed the connector's blanket
# "PCP/DCP > 1 with hybrid attention" refusal to PCP only, because DCP is what
# we needed and PCP is a different sharding we have never measured. #53324
# produces the identical line, so nothing is lost by dropping our version of it:
#
#     if len(kv_cache_config.transfer_groups) > 1 and pcp > 1:
#         unsupported.append(f"PCP > 1 (pcp={pcp}) with hybrid attention")
#
# The verification below asserts both halves of that: the PCP refusal survives
# and the DCP one is gone.
#
# THE KEYSET TEST SWITCHES ITSELF. test_mooncake_dcp_keyset.py checks
# hasattr(MooncakeStoreWorker, "_tail_key_boundaries") and asserts the
# pre-#53324 contract ("the hit is an object boundary") or the #53324 one
# ("_tail_key_boundaries returns, for every group, a boundary whose key the
# store holds"). After this script the second branch is the live one. Nothing
# needs editing there; it is noted because a reader will wonder.
# =============================================================================
set -euo pipefail

echo "=== upstream-53324: swap our Mooncake DCP carry for the merged fix ==="

PATCH=/configs/patches/k3-upstream-53324.patch
SITE=$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin)))")
MC="$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store"

if grep -q "_tail_key_boundaries" "$MC/worker.py"; then
  echo "[upstream-53324] already present in this image"
else
  # Our carry and #53324 rewrite the same functions. If the full v8 patch ran,
  # this would apply on top of it and produce a file that is neither.
  if grep -q "the worker now puts every" "$MC/connector.py"; then
    echo "[upstream-53324] FATAL: k3-ours6-v8-nightly.patch is applied." >&2
    echo "  This chain requires k3-ours6-v8-nomc.patch instead; the two" >&2
    echo "  rewrite the same Mooncake functions and cannot both be present." >&2
    exit 1
  fi
  cd "$SITE"
  patch -p1 --forward --no-backup-if-mismatch --fuzz=0 < "$PATCH"
  echo "[upstream-53324] applied"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
mc = os.path.join(root, "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store")
worker = open(os.path.join(mc, "worker.py")).read()
conn = open(os.path.join(mc, "connector.py")).read()

if "def _tail_key_boundaries" not in worker:
    sys.exit("[upstream-53324] FATAL: _tail_key_boundaries is not defined; "
             "the tail-key resolver is the whole point of #53324")
# The refusal our carry narrowed must survive the swap, and the one it removed
# must stay removed. Getting either backwards silently changes what configs
# this connector accepts.
if "PCP > 1 (pcp=" not in conn:
    sys.exit("[upstream-53324] FATAL: the PCP refusal is gone; PCP sharding is "
             "not mirrored by the block-size scaling and has never been measured")
if "pcp * dcp > 1" in conn:
    sys.exit("[upstream-53324] FATAL: the blanket PCP/DCP refusal is still "
             "present, so DCP with hybrid attention is still refused")

import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker  # noqa: F401
import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector  # noqa: F401
import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator  # noqa: F401

print("[upstream-53324] verified: tail-key resolver present, DCP refusal gone, "
      "PCP refusal kept, modules import")
PY

# kimi-k3-nightly-v8.sh defers this when K3_SKIP_MOONCAKE_CHECKS=1, because at
# that point the tree has neither contract. Now it has #53324's, and the test
# detects which one to assert by looking for _tail_key_boundaries. Running it
# here keeps the guarantee the v8 script gives every other arm: the connector's
# hit-boundary property is checked before any GPU time.
echo "=== mooncake DCP hit-boundary tests (#53324 contract) ==="
python3 /configs/patches/test_mooncake_dcp_keyset.py

echo "=== upstream-53324: done ==="
