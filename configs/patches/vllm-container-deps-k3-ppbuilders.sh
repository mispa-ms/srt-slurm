#!/usr/bin/env bash
# One attention-metadata builder per pipeline stage, instead of one shared by all.
# =============================================================================
# THE BUG. With pipeline parallelism vLLM keeps pp_size batches in flight --
# pp_utils.py seeds its queue with `[None] * get_pp_group().world_size` and consumes
# what step T pushed pp_size steps later. Those in-flight microbatches share one
# attention-metadata builder, because the V2 runner hardcodes both the count and the
# index:
#
#   vllm/v1/worker/gpu/attn_utils.py:144   num_metadata_builders=1
#   vllm/v1/worker/gpu/attn_utils.py:292   attn_group.get_metadata_builder(0)
#
# The builder is not stateless. It owns `self.chunked_prefill_workspace`, the scratch
# buffer the context gather writes into, and a cloned `self._prefill_backend` -- and
# upstream says plainly why that matters:
#
#   "MLA prefill backends keep the prepared metadata on the backend object, so each
#    builder needs its own backend instance to avoid cross-ubatch races."
#
#   "When ubatching is enabled we will have a metadata builder for each ubatch so that
#    if they use internal persistent buffers for cudagraphs, they won't have to worry
#    about conflicting with the other ubatches."
#
# So the race is known and was closed along the DBO/ubatch axis. Pipeline parallelism
# is the same kind of concurrency and was never wired in. Microbatch B's build()
# overwrites the workspace and prepared metadata that microbatch A's kernels are still
# reading, and the read faults.
#
# WHY WE BELIEVE THIS IS OUR IMA. Everything we measured on the current nightly lines
# up with it:
#
#   PP off, DCP8      3628 s, 2231 requests, zero illegal accesses
#   PP2, no-spec      illegal access after a few hundred requests, three times over
#   crash site        moves between the DCP gather, the Mooncake block copy and the
#                     TRTLLM_RAGGED prefill -- whichever kernel is mid-read
#   c96 vs c48        worse at higher concurrency, where overlap and chunking rise
#   -nokvg / -nodirect / #53324  all null: they change which kernel reads, not the sharing
#   pinned 728d3ad    fine, because that image's K3 MLA had no such gather at all
#                     (823 lines against the nightly's 1029)
#
# WHAT THIS CHANGES. Two edits, both in code that already anticipated the problem:
#
#   1. create as many builders as there are pipeline stages
#   2. take the next one per build, round-robin, so consecutive builds -- which under
#      PP are different in-flight microbatches -- never land on the same buffers
#
# Setup-time callers keep index 0 and are untouched; only the per-step build site
# rotates. The cursor lives on the AttentionGroup, so each KV cache group advances
# independently and stays in step with its own builds.
#
# COST. One extra chunked-prefill workspace per pipeline stage. The size is capped at
# 64k tokens (plus 1/dcp for the DCP allgather), 576 columns, bf16 -- about 83 MB per
# builder per MLA group. Against 178 GiB it does not register.
#
# AT pp_size == 1 THIS IS A NO-OP: one builder, and the cursor always returns it.
# =============================================================================
set -euo pipefail

echo "=== ppbuilders: one metadata builder per pipeline stage ==="

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))

# ── 1. AttentionGroup gains a round-robin cursor ──────────────────────────────
utils = os.path.join(root, "vllm/v1/worker/utils.py")
src = open(utils).read()

if "[ppbuilders]" in src:
    print("[ppbuilders] already applied")
    sys.exit(0)

FIELD_ANCHOR = """    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )
"""
if src.count(FIELD_ANCHOR) != 1:
    sys.exit(
        "[ppbuilders] FATAL: expected one metadata_builders field, found %d"
        % src.count(FIELD_ANCHOR)
    )
src = src.replace(
    FIELD_ANCHOR,
    FIELD_ANCHOR
    + """    # [ppbuilders] Round-robin cursor over `metadata_builders`, advanced once per
    # build. Under pipeline parallelism pp_size microbatches are in flight at once
    # and the builder is not stateless -- it owns the chunked-prefill workspace and,
    # for MLA, the prepared metadata on its cloned prefill backend. Handing
    # consecutive builds different builders keeps a later microbatch from
    # overwriting buffers an earlier one's kernels are still reading.
    builder_cursor: int = 0
""",
    1,
)

GETTER_ANCHOR = """    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]
"""
if src.count(GETTER_ANCHOR) != 1:
    sys.exit(
        "[ppbuilders] FATAL: expected one get_metadata_builder, found %d"
        % src.count(GETTER_ANCHOR)
    )
src = src.replace(
    GETTER_ANCHOR,
    GETTER_ANCHOR
    + """
    def next_metadata_builder(self) -> AttentionMetadataBuilder:
        \"\"\"[ppbuilders] The builder for this build, rotating over the pool.

        Callers that only inspect a builder's static properties keep using
        ``get_metadata_builder(0)``; only the per-step build site rotates. With a
        single builder this returns it every time, so pp_size == 1 is unchanged.
        \"\"\"
        builder = self.metadata_builders[self.builder_cursor]
        self.builder_cursor = (self.builder_cursor + 1) % len(self.metadata_builders)
        return builder
""",
    1,
)
compile(src, utils, "exec")
open(utils, "w").write(src)
print("[ppbuilders] applied: " + utils)

# ── 2. the V2 runner creates pp_size of them, and rotates at the build site ────
attn = os.path.join(root, "vllm/v1/worker/gpu/attn_utils.py")
src = open(attn).read()

COUNT_ANCHOR = """                num_metadata_builders=1,
"""
if src.count(COUNT_ANCHOR) != 1:
    sys.exit(
        "[ppbuilders] FATAL: expected one num_metadata_builders=1 in the V2 runner, "
        "found %d" % src.count(COUNT_ANCHOR)
    )
src = src.replace(
    COUNT_ANCHOR,
    """                # [ppbuilders] One per pipeline stage: pp_size microbatches are
                # in flight together and must not share persistent buffers.
                num_metadata_builders=(
                    vllm_config.parallel_config.pipeline_parallel_size
                ),
""",
    1,
)

BUILD_ANCHOR = """            attn_metadata_builder = attn_group.get_metadata_builder(0)
"""
if src.count(BUILD_ANCHOR) != 1:
    sys.exit(
        "[ppbuilders] FATAL: expected one per-step build site, found %d"
        % src.count(BUILD_ANCHOR)
    )
src = src.replace(
    BUILD_ANCHOR,
    """            # [ppbuilders] Rotate: consecutive builds are different in-flight
            # microbatches under PP.
            attn_metadata_builder = attn_group.next_metadata_builder()
""",
    1,
)
compile(src, attn, "exec")
open(attn, "w").write(src)
print("[ppbuilders] applied: " + attn)
PY

# Import the modules rather than trusting that the text compiled, and prove the
# rotation actually rotates -- a cursor that never advances would look identical in a
# log and leave the race exactly where it was.
python3 - <<'PY'
import sys

import vllm.v1.worker.utils as u

g = u.AttentionGroup.__new__(u.AttentionGroup)
g.metadata_builders = ["a", "b"]
g.builder_cursor = 0
seen = [g.next_metadata_builder() for _ in range(4)]
if seen != ["a", "b", "a", "b"]:
    sys.exit("[ppbuilders] FATAL: rotation is wrong: %s" % seen)

g.metadata_builders = ["only"]
g.builder_cursor = 0
if [g.next_metadata_builder() for _ in range(3)] != ["only"] * 3:
    sys.exit("[ppbuilders] FATAL: single-builder case does not return the builder")

import vllm.v1.worker.gpu.attn_utils  # noqa: F401

print("[ppbuilders] verified: rotates over pp_size, no-op at pp_size 1")
PY

echo "=== ppbuilders: done ==="
