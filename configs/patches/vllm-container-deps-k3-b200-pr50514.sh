#!/usr/bin/env bash
# DSpark under PP from vLLM PR #50514 as it stands today, not our old fork of it.
# =============================================================================
# WHY REPLACE OUR OWN PATCH. Our PP support came from this same PR -- the branch
# history says so outright:
#
#   2111011d33  misunp  Adopt upstream PR #50514: spec decode under pipeline parallelism
#
# We took a snapshot, put two K3 fixes on top, and have been hand-rebasing that
# snapshot forward ever since. Meanwhile the PR kept moving, and it did not move in a
# way a rebase can follow: the whole aux-hidden-state handoff was renamed and
# restructured.
#
#   ours                            current PR
#   make_empty_aux_hidden_states    aux_hidden_states_over_pp
#   pack_aux_hidden_states          aux_hidden_state_relay_keys
#   unpack_aux_hidden_states        aux_slot_base / aux_ids
#
# None of our three names appear anywhere in the PR today, and none of them exist on
# upstream main either -- they live only in the image we pinned. Our patch also carries
# a hunk the PR has since dropped, which adds aux recv buffers to
# make_empty_intermediate_tensors under a comment that states its own failure mode:
# "Must match what pack_aux_hidden_states puts on the wire, since the PP transport
# sizes its buffers from this."
#
# That is the shape of the deadlock we measured. Stage 0 sits in
# isend_tensor_dict -> send_object forever while stage 1 has already finished the RPC
# and gone back to its busy loop: the two ends do not agree on what crosses the
# boundary. Our patch is 7 files against the PR's 21, and the gaps are exactly where it
# hurts -- five missing hunks in kimi_k3/nvidia/model.py including both
# _capture_aux_hidden_stream edits ("Fix aux-tap packing under compile and outside a
# workspace"), plus kimi_k3/nvidia/dspark_mla.py and model_executor/models/utils.py,
# which we do not touch at all.
#
# So: stop rebasing, take the PR whole.
#
# WHAT IS IN THE PATCH. The PR's current head, tests removed -- they add nothing at
# runtime and only widen the failure surface. 21 files. Dry-run against a9a17e70, the
# tree this nightly was built from: 0 failed, 0 fuzz, offsets only.
#
# ORDER. Same rule this workstream learned twice the hard way: the strict --fuzz=0 diff
# goes first, on the tree it was derived against, and the content-anchored edits follow
# and tolerate what it moved. Verified after applying the PR for real -- both remaining
# anchors still match exactly once.
#
#   1. PR #50514 head       --fuzz=0 diff
#   2. k3-b200-dcp8.sh      hfshim, PR #53324 guard, DCP dummy-batch fix
#   3. empty_cache          one line before the draft loads
#
# Step 2 touches Mooncake, kv_cache_{coordinator,utils} and model_runner; the PR touches
# none of the first three, so only model_runner is shared and it is content-anchored.
# =============================================================================
set -euo pipefail

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')
command -v patch >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq patch; }

# ── 1. PR #50514, current head ────────────────────────────────────────────────
echo "=== k3-pr50514: spec decode under pipeline parallelism, upstream head ==="
if grep -q "aux_hidden_states_over_pp" "$VLLM_ROOT/vllm/model_executor/models/interfaces.py"; then
    echo "[k3-pr50514] already present; skipping"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 < /configs/patches/k3-pr50514-head.patch > /tmp/pr50514-dry.log 2>&1; then
        echo "[k3-pr50514] FATAL: PR #50514 does not apply to this image" >&2
        cat /tmp/pr50514-dry.log >&2
        exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-pr50514-head.patch
    echo "[k3-pr50514] applied"
fi

python3 - <<'PY'
import importlib.util, os, sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))


def read(p):
    return open(os.path.join(root, p)).read()


# The three things the PR exists to provide, checked by content rather than trusting
# that `patch` returned zero.
if "aux_hidden_states_over_pp" not in read("vllm/model_executor/models/interfaces.py"):
    sys.exit("[k3-pr50514] FATAL: the aux-over-PP interface is missing")
if "pipeline_parallel_size=1" not in read("vllm/config/speculative.py"):
    sys.exit("[k3-pr50514] FATAL: the draft ParallelConfig still inherits target PP")
if "aux_hidden_states_over_pp" not in read("vllm/models/kimi_k3/nvidia/model.py"):
    sys.exit("[k3-pr50514] FATAL: K3 did not opt in to aux-over-PP")
if "aux_hidden_state_relay_keys" not in read("vllm/v1/worker/gpu/pp_utils.py"):
    sys.exit("[k3-pr50514] FATAL: the PP handler has no aux relay keys")
u = read("vllm/v1/worker/gpu/spec_decode/dspark/utils.py")
if "DSpark does not support pipeline parallelism" in u:
    sys.exit("[k3-pr50514] FATAL: the upstream refusal is still present")
print("[k3-pr50514] verified: aux-over-PP present, draft PP pinned, refusal gone")
PY

# ── 2. JET cache, PR #53324 guard, DCP dummy-batch fix ────────────────────────
bash /configs/patches/vllm-container-deps-k3-b200-dcp8.sh

# ── 3. empty_cache before the draft loads ─────────────────────────────────────
echo "=== k3-pr50514: release cached blocks before the draft model ==="

python3 - <<'PY'
import importlib.util, os, sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_runner.py",
)
src = open(target).read()
if "[emptycache]" in src:
    print("[emptycache] already applied: " + target)
    sys.exit(0)

CALL = "self.speculator.load_model(self.model)"
hits = [ln for ln in src.splitlines() if ln.strip() == CALL]
if len(hits) != 1:
    sys.exit(
        "[emptycache] FATAL: found %d lines calling the speculator loader; the load "
        "path moved and this needs re-deriving" % len(hits)
    )

anchor = hits[0]
pad = anchor[: len(anchor) - len(anchor.lstrip())]
addition = "\n".join(
    pad + line if line else ""
    for line in [
        "# [emptycache] The draft's MLA takes a second direct-DCP symmetric-memory",
        "# workspace from the driver, not the caching allocator, and the loader's",
        "# freed staging buffers leave the driver with nothing.",
        "free_before, total = torch.cuda.mem_get_info()",
        "torch.cuda.empty_cache()",
        "free_after, _ = torch.cuda.mem_get_info()",
        "logger.info(",
        '    "[emptycache] driver-visible free before draft load: "',
        '    "%.2f -> %.2f GiB of %.2f GiB",',
        "    free_before / 2**30, free_after / 2**30, total / 2**30,",
        ")",
    ]
) + "\n" + anchor

patched = src.replace(anchor, addition, 1)
compile(patched, target, "exec")
open(target, "w").write(patched)
print("[emptycache] applied at indent %d: %s" % (len(pad), target))
PY

echo "=== k3-pr50514: done ==="
