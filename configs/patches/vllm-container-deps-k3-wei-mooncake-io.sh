#!/usr/bin/env bash
# Wei's two Mooncake I/O flags, and the code that reads them.
# =============================================================================
# WHERE THIS CAME FROM. github.com/wzhao18/vllm, branch wzhao/k3-nvfp4-perf, on
# an 08-26 base. Wei's own summary of it: "One major thing in my branch is that
# it fixes the mooncake kv transfer timeout. But that problem may be specific to
# oci-aga, as there is only one RDMA device." He named exactly two flags to turn
# on, which is what this carries:
#
#     "compact_group_io": true
#     "max_load_batch_keys": 2
#
# in the MooncakeStoreConnector's kv_connector_extra_config.
#
# THREE COMMITS, cherry-picked in order:
#
#   35180218b  Mitigate Mooncake load timeouts with sub-batching
#              Splits one BatchGet into sub-batches of max_load_batch_keys, and
#              -- the part that is a bug fix rather than a knob -- invalidates
#              the blocks of every sub-batch it never attempted after a partial
#              failure. Without that, a timeout leaves later blocks marked
#              loaded when nothing was written into them.
#   a911b24e6  Add compact group-specific Mooncake values
#   ed04c4dbe  Harden compact Mooncake group I/O
#              Pack a transfer group's members into one value instead of one per
#              member, behind compact_group_io. Fewer, larger objects per get.
#
# BOTH FLAGS DEFAULT OFF, in Wei's code and here. Applying this patch changes
# nothing until a config sets them, which is what makes it safe to put in the
# chain and price separately.
#
# WHAT IS NOT HERE, AND WHY. Wei's branch has 28 commits. Six more touch Mooncake
# and four of those collide with our own carry, which reworks the same
# connector/coordinator/worker for DCP hit boundaries:
#
#   a6322e6db  MooncakeStore with hybrid DCP prefix caching   <- ours does this,
#                                                                differently
#   bb9afd157  unselected MultiConnector loads
#   4fce73380  reuse pending load plans
#   2edd47146  k3 hybrid mooncake recompute handling
#
# Merging those two implementations is real work and would put an unmeasured
# rewrite underneath the thing being measured. The three carried here apply
# clean on top of our carry, so this arm prices Wei's flags and only his flags.
#
# THE REST OF HIS BRANCH, checked and deliberately not carried:
#
#   already in our 08-28 image, every added line present:
#     b14c285a3  Release CUDA graph profiling references -- this is upstream
#                #53955, merged between his 08-26 base and our 08-28 nightly
#   net zero -- committed and then reverted on his own branch:
#     16906b4b2 + 849d4c62a   Publish exact recurrent cache boundaries
#     6c0203743 + 280548085   Fix KDA replay convolution history
#   a dead door on our image:
#     e10c92e11  supports_mtp_with_cp_non_trivial_interleave_size on
#                TokenspeedMLAImpl. This is the assert our own interleave carry
#                works around, and a better-shaped fix than ours -- but #50611,
#                which is what raises the interleave above 1, is not in the 08-28
#                image, so the assert cannot fire here and the flag changes
#                nothing. It belongs in UPSTREAM_HANDOFF, not in an arm.
#     3129d926c  Honor DSpark draft load config. Passes
#                speculative_config.draft_load_config into get_model; ours is
#                None, and get_model already does `load_config or
#                vllm_config.load_config`, so the call is identical for us.
#   a different axis, not this arm:
#     d003d2177, 895b9f046, 0df0a7813   NVFP4 -- we are MXFP4 on B300
#     57b34cdc2, 7677538f9              NIXL heterogeneous DCP -- disagg
#     19a843c00, f8dad1d92              routing diagnostics behind an env var
#     4060d94ce, 48be0d424, 37da1cc63   MoE buckets, in_proj align, DFlash
#     ca46b8b12, 1c99b6497, 27ab92bd4   documentation only
#
# WHY THIS IS WORTH AN ARM HERE AND NOT ONLY ON oci-aga. Wei suspects the timeout
# is specific to a single-RDMA-device cluster. Our reason is different and does
# not depend on his being right: our Mooncake external hit rate is the standing
# gap against the AMD curve -- LMCache 85.5% against our 9.4% on the run that
# lost -- and both of these flags act on exactly that path.
# =============================================================================
set -euo pipefail

echo "=== wei-mooncake-io: compact_group_io + max_load_batch_keys ==="

PATCH=/configs/patches/k3-wei-mooncake-io.patch
SITE=$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin)))")

cd "$SITE"
if grep -q "compact_group_io" vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py; then
  echo "[wei-mooncake-io] already present in this image"
else
  # -p1 against the site-packages root, the same way every other patch in this
  # directory is applied. Refuse rather than fuzz: these hunks sit next to our
  # own carry's hunks in the same two files, and a fuzzy apply there would land
  # somewhere plausible and wrong.
  patch -p1 --forward --no-backup-if-mismatch --fuzz=0 < "$PATCH"
  echo "[wei-mooncake-io] applied"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
worker = os.path.join(
    root, "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py")
src = open(worker).read()

# Both flags must be READ FROM THE CONFIG, not merely defined. A patch that
# lands the helpers but misses the extra_config lookup would leave the flags
# silently inert, which is the failure this check exists for.
for needle, what in (
    ('extra_config.get("compact_group_io"', "compact_group_io lookup"),
    ('extra_config.get("max_load_batch_keys")', "max_load_batch_keys lookup"),
    ("_compact_group_io_schema_fingerprint", "compact group schema fingerprint"),
    ("self.max_load_batch_keys", "sub-batch size on the recv thread"),
):
    if needle not in src:
        sys.exit(f"[wei-mooncake-io] FATAL: {what} missing after applying")

# Both must default OFF, so that this patch in the chain is a no-op until a
# config asks for it and the arm that measures it is the only thing that differs.
if 'extra_config.get("compact_group_io", "False")' not in src:
    sys.exit("[wei-mooncake-io] FATAL: compact_group_io no longer defaults False")

import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker  # noqa: F401
import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data  # noqa: F401

print("[wei-mooncake-io] verified: both flags read from extra_config, "
      "both default off, modules import")
PY

echo "=== wei-mooncake-io: done ==="
