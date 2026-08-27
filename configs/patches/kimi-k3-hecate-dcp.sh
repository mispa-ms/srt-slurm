#!/usr/bin/env bash
# Kimi-K3 on Hecate / VR200 with DCP: the bring-up probe, then remove the one
# assert in the Rubin image's plugin that refuses decode context parallelism.
#
# ── WHAT THE ASSERT IS ───────────────────────────────────────────────────────
# Pipeline 64903207 died in every rank with
#
#   File ".../vllm/models/kimi_k3/nvidia/mla.py", line 309, in __init__
#     parallel_config.decode_context_parallel_size <= 1
#   AssertionError: Kimi-K3 MultiHeadLatentAttention does not support context
#   parallelism.
#
# It is NOT upstream. Checked against two other builds of the same file:
#   vLLM main                     no such assert
#   container sha 46638857f (0.26.1)  no such assert  <- what the B300 AGG v7
#                                     ladder runs, at DCP8, for 9,889 tok/s/GPU
# Both carry the full implementation the assert would forbid -- MLADCPManager is
# imported at mla.py:95, dcp_world_size is read at :347, the manager is built at
# :356, and query_gather is used in the decode path at :643.
#
# So this is a guard added on the NVIDIA rubin branch over an implementation that
# exists and is exercised elsewhere, not a stub raising because nothing is there.
#
# ── WHY THIS IS STILL A RISK ─────────────────────────────────────────────────
# The other reading is that the rubin branch refactored MLA and DCP genuinely
# broke, and the assert was added to stop it producing wrong output. Removing an
# assert cannot distinguish those two: if the second is true, this patch turns a
# crash into silently wrong attention, which is worse than not running.
#
# This script therefore refuses to remove the assert unless the DCP
# implementation it guards is present and intact, and prints what it found so
# the decision is auditable from the log. A PERFORMANCE RUN CANNOT CONFIRM
# CORRECTNESS -- pair this with an accuracy arm before trusting any number.

set -euo pipefail

bash /configs/patches/kimi-k3-hecate-probe.sh

echo "=== removing the K3 decode-context-parallel assert ==="

python3 - <<'PY'
import pathlib
import re
import sys

import vllm

target = pathlib.Path(vllm.__file__).parent / "models/kimi_k3/nvidia/mla.py"
if not target.is_file():
    sys.exit(f"FATAL: {target} not found; the K3 plugin layout changed.")
src = target.read_text()
MARK = "PATCHED-kimi-k3-hecate-dcp"

# 1. The implementation the assert guards must actually be here. If the rubin
#    branch stripped DCP out and left the assert as the only trace, removing it
#    would produce wrong output rather than a crash.
NEEDED = {
    "MLADCPManager import": r"from vllm\.v1\.attention\.ops\.dcp import .*MLADCPManager",
    "dcp_world_size read": r"self\.dcp_world_size\s*=\s*parallel_config\.decode_context_parallel_size",
    "manager construction": r"self\.dcp_manager\s*=\s*MLADCPManager\(",
    "decode-path query_gather": r"self\.dcp_manager\.query_gather\(",
}
missing = [name for name, rx in NEEDED.items() if not re.search(rx, src)]
if missing:
    sys.exit("FATAL: the DCP implementation is not intact in this build: "
             f"missing {missing}. The assert is guarding a real hole -- do NOT "
             "remove it. Run without DCP instead.")
print("DCP implementation present:")
for name in NEEDED:
    print(f"  ok  {name}")

if MARK in src:
    print("already patched, skipping")
else:
    # 2. Remove only the assert, located with the AST rather than a regex.
    #    The rubin build spells it across three lines --
    #        assert (
    #            parallel_config.decode_context_parallel_size <= 1
    #        ), "Kimi-K3 MultiHeadLatentAttention does not support context parallelism."
    #    -- which a single-line pattern silently misses. Matching the parsed
    #    statement is formatting-proof.
    import ast

    tree = ast.parse(src)
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.Assert)
            and isinstance(getattr(n, "msg", None), ast.Constant)
            and isinstance(n.msg.value, str)
            and "does not support context parallelism" in n.msg.value]
    if len(hits) != 1:
        ctx = [f"{i+1}: {l}" for i, l in enumerate(src.splitlines())
               if "context parallelism" in l]
        sys.exit(f"FATAL: expected exactly one such assert, found {len(hits)}. "
                 f"Lines mentioning context parallelism: {ctx}")
    node = hits[0]
    lines = src.splitlines(keepends=True)
    lo, hi = node.lineno - 1, node.end_lineno            # 0-based, exclusive
    indent = len(lines[lo]) - len(lines[lo].lstrip())
    pad = " " * indent
    replacement = (f"{pad}# {MARK}: assert removed. It is not upstream and not in\n"
                   f"{pad}# the 0.26.1 plugin the B300 DCP8 ladder runs; the\n"
                   f"{pad}# implementation it guards was verified present above.\n")
    print(f"removing assert at lines {node.lineno}-{node.end_lineno}:")
    for l in lines[lo:hi]:
        print("   -", l.rstrip())
    target.write_text("".join(lines[:lo]) + replacement + "".join(lines[hi:]))
    ast.parse(target.read_text())          # never leave the file unparseable
    print(f"patched {target}")

# 3. Say out loud what the file now does, so the log carries the evidence.
after = target.read_text()
print("post-patch check:")
print("  assert present :", "does not support context parallelism" in after)
print("  MLADCPManager  :", "MLADCPManager(" in after)
PY

echo "=== assert removed; DCP is now reachable ==="
echo "NOTE: this is a performance path only. Nothing here proves DCP produces"
echo "      correct attention on this branch -- an accuracy arm has to say that."
