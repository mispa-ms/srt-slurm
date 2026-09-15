#!/usr/bin/env python3
"""Fence after each breakable CUDA-graph segment, on a switch.

WHOSE RESULT THIS TESTS. The owners measured that synchronising after every
breakable graph / eager segment completes a full one-hour profile -- 3,050
requests, zero stalls, Xids or launch failures, ~12.3k tok/s/GPU, within 3% of
the bia reference. That is their finding, not ours, and their own summary says
the kernel-level root cause is still unproven. This script reproduces the
mitigation on aws-pdx, where it has never been run: our c70 stalls 20 of 24, so
three clean hours here would be p ~ 0.005 and an independent confirmation on a
second cluster.

WHAT IT CHANGES. `BreakableCUDAGraphCapture.replay()` is a bare loop over the
recorded segments. The patch inserts one fence per segment:

    for r in self.segments:
        r()
        torch.cuda.current_stream().synchronize()     # <- added

MODES (env `PDX_SEGSYNC`):
    all       fence after every segment -- the owners' configuration
    graph     fence only after graph segments, not after eager ones. This is
              the experiment the owners are running next; do not spend runs on
              it here.
    depth:N   fence only when more than N segments are outstanding. Nobody has
              run this. Changing the driver's launch-queue size was already
              tried and did not help, but capping OUTSTANDING SEGMENTS is a
              different quantity: if depth:8 holds, the cause is queue depth;
              if only depth:1 holds, it is a per-segment ordering dependency.

Unset or empty leaves the file untouched, so the control arm is byte-identical.

Diagnostics only: it must never fail a run. Any problem is reported and the
script exits 0 with the file unchanged.
"""

import os
import re
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/compilation/breakable_cudagraph.py"
)
MARKER = "# pdx-segsync"


def main() -> int:
    mode = os.environ.get("PDX_SEGSYNC", "").strip()
    if not mode:
        print("    segsync: off (set PDX_SEGSYNC=all|graph|depth:N to enable)")
        return 0

    if mode != "all" and mode != "graph" and not re.fullmatch(r"depth:\d+", mode):
        print(f"    segsync: WARNING unrecognised PDX_SEGSYNC={mode!r}; leaving "
              f"the file untouched")
        return 0

    path = os.environ.get("PDX_SEGSYNC_TARGET", TARGET)
    try:
        src = open(path).read()
    except OSError as e:
        print(f"    segsync: WARNING cannot read {path}: {e}")
        return 0

    if MARKER in src:
        print("    segsync: already applied")
        return 0

    # Match the loop body exactly rather than a line number: the file is the
    # image's, and a line number quoted from a different checkout is how this
    # investigation has already misread code once.
    pat = re.compile(
        r"(?P<head>\n(?P<ind>[ ]+)def replay\(self\)[^\n]*:\n"
        r"(?:(?P=ind)[ ]+\"\"\".*?\"\"\"\n)?"
        r")(?P<loop>(?P=ind)[ ]+for r in self\.segments:\n"
        r"(?P=ind)[ ]{8}r\(\)\n)",
        re.S,
    )
    m = pat.search(src)
    if not m:
        print("    segsync: WARNING replay() loop not found in the expected "
              "shape; leaving the file untouched")
        return 0

    ind = m.group("ind")
    body = ind + " " * 4 + "for i, r in enumerate(self.segments):\n"
    body += ind + " " * 8 + "r()\n"
    if mode == "all":
        cond = None
    elif mode == "graph":
        # Eager segments are plain callables recorded by add_eager; graph
        # segments are bound `replay` methods of a cudagraph object.
        cond = "if getattr(r, '__self__', None) is not None:"
    else:
        n = int(mode.split(":", 1)[1])
        cond = f"if (i + 1) % {n} == 0:"

    if cond is None:
        body += ind + " " * 8 + f"torch.cuda.current_stream().synchronize()  {MARKER} {mode}\n"
    else:
        body += ind + " " * 8 + f"{cond}  {MARKER} {mode}\n"
        body += ind + " " * 12 + "torch.cuda.current_stream().synchronize()\n"

    out = src[: m.start("loop")] + body + src[m.end("loop") :]
    if "import torch" not in out.split("def ", 1)[0]:
        print("    segsync: WARNING torch is not imported at module scope; "
              "leaving the file untouched")
        return 0

    try:
        open(path, "w").write(out)
    except OSError as e:
        print(f"    segsync: WARNING cannot write {path}: {e}")
        return 0

    print(f"    segsync: applied mode={mode} to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
