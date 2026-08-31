#!/usr/bin/env bash
# The DCP branch of the MLA builder substitutes a tensor the dummy batch leaves None.
# =============================================================================
# THE CRASH. Every node of a DP x PP x DCP aggregate died at the same instant:
#
#   gpu_worker.execute_dummy_batch -> model_runner._dummy_run(uniform_decode=True)
#     -> mamba_hybrid.prepare_attn -> attn_utils.build_attn_metadata
#     -> mla_attention.build_for_cudagraph_capture -> self.build(0, m)
#     -> seq_lens_device=seq_lens[:num_decodes]
#   TypeError: 'NoneType' object is not subscriptable
#
# THREE THINGS HAVE TO BE TRUE AT ONCE, which is why nothing hit it before:
#
#   1. DATA PARALLELISM. execute_dummy_batch exists only for DP: with MoE, a rank
#      holding no requests must still run an aligned forward so the expert
#      collectives match. Without DP that path is never called. Every K3 arm on
#      this workstream before this one was DP-free.
#   2. DCP > 1. The branch that substitutes is guarded on dcp_world_size > 1.
#   3. MLA WITH FULL CUDAGRAPH CAPTURE, which is where build() is reached with a
#      dummy batch rather than a real one.
#
# THE DEFECT. InputBatch's dummy constructor sets the field to None outright --
# `dcp_local_seq_lens=None`, under a comment reading "Dummy: seq_len == query_len
# (fresh-prefill shape)" -- while the real InputBatch allocates it as
# `torch.zeros(max_num_reqs, dtype=torch.int32, device=device)`. The MLA builder
# then does:
#
#     if self.dcp_world_size > 1:
#         dcp_tot_seq_lens_device = seq_lens[:num_decodes]
#         seq_lens = dcp_local_seq_lens          # <- None on the dummy path
#         ...
#     decode_metadata = self._build_decode(
#         seq_lens_device=seq_lens[:num_decodes],   # <- raises here
#
# It replaces seq_lens unconditionally with a field its caller is allowed to omit.
#
# THE FIX, AND WHY THIS FORM. Substitute only when there is something to
# substitute. The two tensors are the same shape and dtype -- per-request int32
# over max_num_reqs -- so falling back to the global seq_lens gives the capture
# the shape it needs, and a capture depends on shapes, not on the values. On a
# real DCP decode dcp_local_seq_lens is always populated, so the fallback never
# fires and nothing measured changes.
#
# max_seq_len keeps its DCP adjustment either way: that arithmetic sizes the
# workspace for a rank's share and is right regardless of which seq_lens tensor
# is in hand. Guarding it too would silently over-allocate the captured graph.
#
# THE OTHER FIX, for whoever upstreams this. The dummy constructor could allocate
# the tensor like the real one does, which is arguably the more correct place --
# every other consumer assumes the field is there when DCP is on. That is a larger
# change to a constructor shared by every backend, so it is not what we carry.
# Say both in the PR and let a reviewer choose.
# =============================================================================
set -euo pipefail

echo "=== dcp-dummy-seqlens: do not substitute a tensor the dummy batch omits ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/model_executor/layers/attention/mla_attention.py",
)
src = open(target).read()

FIXED = "if dcp_local_seq_lens is not None:"
if FIXED in src:
    print("[dcp-dummy-seqlens] already present in this image")
    sys.exit(0)

# Anchor on the substitution and its indent rather than a line number: this file
# is 2,500 lines and moves between nightlies.
ANCHOR = "                seq_lens = dcp_local_seq_lens\n"
if src.count(ANCHOR) != 1:
    sys.exit(
        "[dcp-dummy-seqlens] FATAL: expected one 'seq_lens = dcp_local_seq_lens' at "
        "16-space indent, found %d -- the builder was restructured and this patch "
        "needs re-deriving" % src.count(ANCHOR)
    )

REPLACEMENT = (
    "                # The dummy batch execute_dummy_batch builds under DP leaves\n"
    "                # this field None, and substituting it would hand the decode\n"
    "                # builder a None to slice. Both tensors are per-request int32\n"
    "                # over max_num_reqs, so keeping the global one gives a capture\n"
    "                # the shape it needs; on a real decode the local tensor is\n"
    "                # always populated and this is a no-op.\n"
    "                if dcp_local_seq_lens is not None:\n"
    "                    seq_lens = dcp_local_seq_lens\n"
)
src = src.replace(ANCHOR, REPLACEMENT, 1)
compile(src, target, "exec")
open(target, "w").write(src)
print("[dcp-dummy-seqlens] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import re
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path = os.path.join(root, "vllm/model_executor/layers/attention/mla_attention.py")
src = open(path).read()

if "if dcp_local_seq_lens is not None:" not in src:
    sys.exit("[dcp-dummy-seqlens] FATAL: the guard is not in the file after writing it")
# The unguarded form must be gone, or both exist and the first still crashes.
if re.search(r"^ {16}seq_lens = dcp_local_seq_lens$", src, re.M):
    sys.exit("[dcp-dummy-seqlens] FATAL: an unguarded substitution is still present")
# The workspace sizing must NOT have been swept into the guard.
if not re.search(r"^ {16}num_partitions = ", src, re.M):
    sys.exit(
        "[dcp-dummy-seqlens] FATAL: the max_seq_len adjustment moved indent; it must "
        "stay outside the guard or the captured graph is sized wrong"
    )

import vllm.model_executor.layers.attention.mla_attention  # noqa: F401

print("[dcp-dummy-seqlens] verified: guard present, no unguarded form, "
      "workspace sizing untouched")
PY

echo "=== dcp-dummy-seqlens: done ==="
