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
    probe     **does not fix anything.** Records one event per segment and no
              synchronize, so the timing is left alone and the run is expected
              to stall exactly as the control does. A daemon thread samples the
              events once a second and logs which segment the device last
              finished, so the stall names the node it stopped at.

WHY `probe` RATHER THAN cuda-gdb. The device-side question -- which node did the
front end stop fetching at -- is what cuda-gdb would answer, but cuda-gdb is not
in this image and is reported not to initialise against an already-wedged
context. A host-readable progress counter answers the same question without a
debugger, which is how the owners' `device completed N/M, FIRST PENDING #k`
numbers are produced. The monitor has to be a separate thread: the replaying
thread is inside a CUDA call when it wedges and will never run Python again.

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

PROBE_TAIL = '''

# --- pdx-segsync probe ------------------------------------------------------
# Timing-neutral: one event recorded per segment, never waited on. A daemon
# thread samples them, so when the replaying thread wedges inside a CUDA call
# the last sample still names the segment the device finished last.
import threading as _pdx_threading
import time as _pdx_time

_PDX_PROBE_EVENTS: dict = {}
_PDX_PROBE_STATE: dict = {}
_PDX_PROBE_LOCK = _pdx_threading.Lock()
_PDX_PROBE_STARTED = False


def _pdx_probe_record(capture, i):
    try:
        key = id(capture)
        ring = _PDX_PROBE_EVENTS.get(key)
        if ring is None:
            ring = _PDX_PROBE_EVENTS[key] = {}
        ev = ring.get(i)
        if ev is None:
            ev = ring[i] = torch.cuda.Event()
        ev.record()
        _PDX_PROBE_STATE[key] = (i, len(capture.segments), _pdx_time.time())
        _pdx_probe_start()
    except Exception:
        pass


def _pdx_probe_loop():
    import logging

    log = logging.getLogger("pdx.segprobe")
    while True:
        _pdx_time.sleep(1.0)
        try:
            for key, (last, total, ts) in list(_PDX_PROBE_STATE.items()):
                ring = _PDX_PROBE_EVENTS.get(key) or {}
                pending = None
                done = 0
                for i in sorted(ring):
                    if ring[i].query():
                        done += 1
                    elif pending is None:
                        pending = i
                age = _pdx_time.time() - ts
                log.warning(
                    "segprobe capture=%x segments=%d issued_through=%d "
                    "device_completed=%d first_pending=%s issue_age=%.1fs",
                    key, total, last, done,
                    "none" if pending is None else pending, age,
                )
        except Exception:
            pass


def _pdx_probe_start():
    global _PDX_PROBE_STARTED
    if _PDX_PROBE_STARTED:
        return
    with _PDX_PROBE_LOCK:
        if _PDX_PROBE_STARTED:
            return
        _PDX_PROBE_STARTED = True
        t = _pdx_threading.Thread(
            target=_pdx_probe_loop, name="pdx-segprobe", daemon=True
        )
        t.start()
'''


def _apply_coulten(path, src):
    """The owners' spec, implemented literally rather than approximated.

    From their write-up: explicit GRAPH/EAGER types recorded at append time so
    callback and metadata cannot drift; synchronize the replay stream after
    every GRAPH callback INCLUDING the terminal one; never after an EAGER
    callback; fail before replay if the two lists disagree.

    Our earlier `all` mode inferred the type from `__self__` and fenced both
    kinds. That is a superset of their sync points, so it was not the reason
    our run stalled -- the reason was that under FULL_AND_PIECEWISE a
    FULL-dispatched step returns the decorated fn unwrapped and never enters
    replay() at all. Recording the type at the append site removes the
    inference; it does not remove that.
    """
    # 1. Tag the graph segment where it is recorded.
    old_graph = "        self.segments.append(self._current_graph.replay)\n"
    new_graph = ("        self._pdx_append(self._current_graph.replay, True)"
                 "  " + MARKER + " graph\n")
    # 2. Tag the eager segment where it is recorded.
    old_eager = "        self.segments.append(fn)\n"
    new_eager = ("        self._pdx_append(fn, False)  " + MARKER + " eager\n")
    # 3. Replace the replay loop.
    old_replay = ("    def replay(self) -> None:\n"
                  "        for r in self.segments:\n"
                  "            r()\n")
    new_replay = (
        "    def _pdx_append(self, cb, is_graph):  " + MARKER + "\n"
        "        types = getattr(self, '_pdx_segment_types', None)\n"
        "        if types is None:\n"
        "            types = self._pdx_segment_types = []\n"
        "        self.segments.append(cb)\n"
        "        try:\n"
        "            types.append(is_graph)\n"
        "        except BaseException:\n"
        "            self.segments.pop()\n"
        "            raise\n"
        "\n"
        "    def replay(self) -> None:\n"
        "        types = getattr(self, '_pdx_segment_types', None) or []\n"
        "        if len(types) != len(self.segments):\n"
        "            raise RuntimeError('pdx-segsync: segment metadata is inconsistent')\n"
        # current_stream() is re-read each iteration, as in their pseudocode:
        # an eager segment could in principle change the current stream, and
        # hoisting the lookup would then fence the wrong one.
        "        for r, is_graph in zip(self.segments, types):\n"
        "            r()\n"
        "            if is_graph:\n"
        "                torch.cuda.current_stream().synchronize()\n")
    for old in (old_graph, old_eager, old_replay):
        if old not in src:
            print("    segsync: WARNING coulten mode could not find one of its "
                  "three anchors; leaving the file untouched")
            return 0
    out = src.replace(old_graph, new_graph, 1)
    out = out.replace(old_eager, new_eager, 1)
    out = out.replace(old_replay, new_replay, 1)
    return _write(path, out, "coulten")


def _write(path, out, mode):
    try:
        open(path, "w").write(out)
    except OSError as e:
        print(f"    segsync: WARNING cannot write {path}: {e}")
        return 0
    print(f"    segsync: applied mode={mode} to {path}")
    return 0


def main() -> int:
    mode = os.environ.get("PDX_SEGSYNC", "").strip()
    if not mode:
        print("    segsync: off (set PDX_SEGSYNC=all|graph|depth:N to enable)")
        return 0

    if mode not in ("all", "graph", "probe", "coulten") and not re.fullmatch(r"depth:\d+", mode):
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

    if mode == "coulten":
        return _apply_coulten(path, src)

    ind = m.group("ind")
    body = ind + " " * 4 + "for i, r in enumerate(self.segments):\n"
    body += ind + " " * 8 + "r()\n"
    if mode == "probe":
        body += ind + " " * 8 + f"_pdx_probe_record(self, i)  {MARKER} probe\n"
        out = src[: m.start("loop")] + body + src[m.end("loop") :] + PROBE_TAIL
        return _write(path, out, mode)
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
