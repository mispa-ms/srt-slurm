#!/usr/bin/env bash
# Wei Zhao's Mooncake stack, on top of the 08/28 DSpark recipe.
# =============================================================================
# Source: https://github.com/wzhao18/vllm/tree/wzhao/k3-nvfp4-perf, six commits
# on a 2026-08-26 base (080a66a6):
#
#   2edd471469  fix k3 hybrid mooncake recompute handling
#   bb9afd1578  Fix unselected Mooncake MultiConnector loads
#   35180218b5  Mitigate Mooncake load timeouts with sub-batching
#   a911b24e60  Add compact group-specific Mooncake values
#   ed04c4dbec  Harden compact Mooncake group I/O
#   4fce733806  Reuse pending Mooncake load plans
#
# A seventh commit on that branch, a6322e6db5 "Support MooncakeStore with hybrid
# DCP prefix caching", is NOT here: it is the same change as vLLM #53324, which
# vllm-container-deps-k3-b200-dcp8.sh already applies. The two differ only in a
# field upstream renamed between 08/26 and 08/28 -- kv_cache_groups became
# transfer_groups -- which is why this patch is cut against the 08/28 image
# (6f7df92a) with #53324 already on it, and will not apply anywhere else.
#
# One hunk needed hand resolution. bb9afd1578 drops the `num_external_tokens > 0`
# guard in MooncakeStoreScheduler.update_state_after_alloc, because MultiConnector
# passes zero when a different child was selected to load and the guard then threw
# the blocks away so they were never stored. Wei's base has no #53324, so his line
# is a bare blocks.get_block_ids(); ours has to keep the select_transfer_block_ids()
# wrapper #53324 introduced. The merged form drops the guard and keeps the wrapper.
#
# WHAT THIS BUYS IS NOT SETTLED, AND THE FLAGS ARE THE EXPERIMENT.
# Both knobs default to off (compact_group_io -> "False", max_load_batch_keys ->
# None), so applying this patch alone should be behaviour-neutral except for the
# three commits that change unconditionally. Set them in kv_connector_extra_config:
#
#   "compact_group_io": true      separates block stride from transfer length, so a
#                                 block moves only its real bytes instead of a full
#                                 padded stride. K3 is hybrid -- MambaSpec groups
#                                 beside attention groups, with very different block
#                                 sizes -- so this is the one with a mechanism that
#                                 plausibly applies to us.
#   "max_load_batch_keys": 2      splits BatchGet into pairs. Wei added it for
#                                 oci-aga, which has a single RDMA device; he said
#                                 himself the timeout may be specific to that. prenyx
#                                 has eight NICs and our 08/28 AGG logs show no
#                                 timeout, no BatchGet failure and no partial-failure
#                                 invalidation, so on this cluster sub-batching is
#                                 expected to cost rather than pay. It is measured
#                                 rather than assumed.
# =============================================================================
set -euo pipefail

echo "=== wei-mooncake: Wei's six Mooncake commits ==="

bash /configs/patches/vllm-container-deps-k3-b200-828-dspark3.sh

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
STORE="$VLLM_ROOT/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store"

if [ ! -d "$STORE" ]; then
    echo "[wei-mooncake] FATAL: no mooncake store package in this image" >&2
    exit 1
fi

if grep -q "max_load_batch_keys" "$STORE/worker.py"; then
    echo "[wei-mooncake] already applied to this image"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 \
         < /configs/patches/k3-mooncake-wei6-828.patch > /tmp/wei-mooncake-dry.log 2>&1; then
        echo "[wei-mooncake] FATAL: does not apply to this image" >&2
        echo "[wei-mooncake] the patch is cut against 6f7df92a + #53324; check both" >&2
        cat /tmp/wei-mooncake-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-mooncake-wei6-828.patch
    echo "[wei-mooncake] applied under $VLLM_ROOT"
fi

# Verify what the arms actually depend on: that both knobs are read from
# kv_connector_extra_config, that they still default to off, and that the
# hand-resolved hunk kept #53324's wrapper. A silent no-op here would make every
# arm below measure the control twice.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
store = os.path.join(root, "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store")
worker = open(os.path.join(store, "worker.py")).read()
sched = open(os.path.join(store, "scheduler.py")).read()

for knob in ("compact_group_io", "max_load_batch_keys"):
    if f'extra_config.get("{knob}"' not in worker:
        sys.exit(f"[wei-mooncake] FATAL: {knob} is not read from kv_connector_extra_config")

if 'extra_config.get("compact_group_io", "False")' not in worker:
    sys.exit("[wei-mooncake] FATAL: compact_group_io no longer defaults to off")

if "self.block_stride" not in worker and "block_stride" not in open(os.path.join(store, "data.py")).read():
    sys.exit("[wei-mooncake] FATAL: compact group I/O did not bring block_stride")

if "select_transfer_block_ids" not in sched:
    sys.exit("[wei-mooncake] FATAL: the merged hunk dropped #53324's select_transfer_block_ids()")
if "if num_external_tokens > 0:" in sched.split("local_block_ids")[0][-400:]:
    sys.exit("[wei-mooncake] FATAL: the num_external_tokens guard survived the merge")

print("[wei-mooncake] verified: both knobs parsed, default off, #53324 wrapper kept")
PY

echo "=== wei-mooncake: done ==="
