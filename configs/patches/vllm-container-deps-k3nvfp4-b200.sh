#!/usr/bin/env bash
# Kimi-K3-NVFP4 on B200: JET checkpoint plus the DCP dummy-batch fix.
# =============================================================================
# The NVFP4 counterpart of vllm-container-deps-k3-b200-dcp8.sh. Two things before the
# server starts, and two deliberate omissions.
#
#   1. vllm-container-deps-k3nvfp4-hfshim.sh, which resolves nvidia/Kimi-K3-NVFP4 to
#      the JET copy mounted at /jet-k3nvfp4 instead of pulling 1.6 TB inside the job.
#
#   2. The DCP dummy-batch fix, inlined below. make_dummy() leaves
#      dcp_local_seq_lens unset while the MLA metadata builder swaps it in for
#      seq_lens whenever DCP is on, so a DP-idle step reads a stale buffer.
#
# WHY NOT JUST CALL THE MXFP4 SCRIPT. vllm-container-deps-k3-b200-dcp8.sh carries the
#   same fix, but it opens by chaining vllm-container-deps-k3-hfshim.sh, which wires the
#   MXFP4 cache and exits 1 when its staged directory is absent. On an NVFP4 recipe
#   K3_STAGED_DIR is not set, the default path does not exist on prenyx, and the job
#   would die before the engine started. The fix is 30 lines; the coupling is not worth
#   untangling for one arm family.
#
# WHERE EACH IMAGE STANDS, checked rather than assumed:
#
#   patch                      728d3ad (Xin's branch)   nightly @ main
#   NVFP4 hfshim               needed                   needed
#   dcp_local_seq_lens dummy   already present          absent, applied here
#
#   Both paths are idempotent -- the marker check below prints and exits rather than
#   patching twice -- so the same script is correct on either image.
#
# NOT INCLUDED: vLLM PR #53324, MooncakeStore with hybrid DCP prefix caching. The GB300
#   NVFP4 arm carries it because Wei Zhao runs it for nvfp4 + trtllm-moe + dep16, but the
#   diff was taken against his base and does not apply to this image -- the first attempt
#   died with "no applier succeeded (git apply, patch, patch-ng)" against
#   0.1.dev19908+g728d3ad09. It is also not something this stack has been shown to need:
#   every MXFP4 arm on this workstream runs Mooncake with DCP8 without it. If NVFP4
#   surfaces the Mooncake -7 the PR fixes, re-derive the diff against the image actually
#   in use and add it back here.
#
# NOT INCLUDED: the empty_cache() before the draft load. That is a DSpark requirement
#   and these are no-spec arms; nothing builds a drafter, so the symm_mem workspace it
#   guards is never allocated. Add the chain back when the DSpark NVFP4 arms go out.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3nvfp4-hfshim.sh

echo "=== k3nvfp4-b200: applying the DCP dummy-batch fix ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_runner.py",
)
src = open(target).read()

marker = "input_batch.dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens"
if marker in src:
    print(f"[k3nvfp4-b200] already applied: {target}")
    sys.exit(0)

anchor = """                max_query_len=batch_desc.max_query_len,
            )
            if not skip_attn_for_dummy_run:"""
if src.count(anchor) != 1:
    sys.exit(
        f"[k3nvfp4-b200] FATAL: anchor found {src.count(anchor)} times in {target}; "
        "the dummy-batch path moved and this patch needs re-deriving"
    )

addition = """                max_query_len=batch_desc.max_query_len,
            )
            # Same DCP handling as create_dummy_attn_state: make_dummy leaves
            # dcp_local_seq_lens unset, but the MLA metadata builder swaps it
            # in for seq_lens whenever DCP is enabled.
            if self.use_dcp:
                prepare_dcp_local_seq_lens(
                    self.input_buffers.dcp_local_seq_lens,
                    input_batch.seq_lens,
                    dummy_num_reqs,
                    self.dcp_size,
                    self.dcp_rank,
                    self.cp_interleave,
                )
                input_batch.dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[
                    :dummy_num_reqs
                ]
            if not skip_attn_for_dummy_run:"""

patched = src.replace(anchor, addition)
compile(patched, target, "exec")
open(target, "w").write(patched)
print(f"[k3nvfp4-b200] applied: {target}")
PY

echo "=== k3nvfp4-b200: done ==="
