#!/usr/bin/env python3
"""Reuse one CUDA event per rank in the Mooncake store's save fence.

WHY THIS AND NOT THE RING. `pdx_store_event_pool.py` kept each per-step event
alive so the sending thread would stop calling `cudaEventDestroy`. That arm ran
(431607) and stalled at +37.8 min against the control's +38.1 min, so the
off-thread destroy is not the cause. But it moved only the *destroy*: the
connector still calls `torch.cuda.Event()` -- and therefore
`cudaEventCreateWithFlags` -- once per saved step, and once the ring fills the
destroys resume on the main thread.

This patch removes the churn entirely. One event is created the first time a step
has something to save and re-recorded on every step after, so across a whole run
there is exactly one create and no destroy.

It is the last untested element of the save path. The others are closed:

    record on the compute stream   structural, cannot be removed
    synchronize on a worker thread `-blockev` arm, Event(blocking=True): stalled
    destroy on a worker thread     `-evtpool` arm, event ring:            stalled
    the device copy itself         b300-dsxe's 0.3.13.post1+copystream:   2/2 stall
    RDMA read of GPU memory        their tcp arms stall too, so not required

SEMANTICS. The sending thread waits on whatever the event last recorded. If the
main thread re-records while a worker is inside `synchronize()`, that worker
waits for a later step's compute than it needed -- longer, never shorter, so the
fence it is providing still holds.
"""

import ast
import pathlib
import sys

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/"
    "kv_connector/v1/mooncake/store/worker.py"
)

OLD = """            current_event = None
            for request in meta.requests:
                if request.can_save:
                    current_event = torch.cuda.Event()
                    current_event.record()
                    break
"""

NEW = """            current_event = None
            for request in meta.requests:
                if request.can_save:
                    # PDX: one event per rank, created once and re-recorded, so
                    # no cudaEventCreate/cudaEventDestroy runs per saved step on
                    # any thread. Waiting on a later record is longer, never
                    # shorter, so the fence still holds.
                    current_event = getattr(self, "_pdx_reused_event", None)
                    if current_event is None:
                        current_event = torch.cuda.Event()
                        self._pdx_reused_event = current_event
                    current_event.record()
                    break
"""


def main() -> int:
    if not TARGET.exists():
        print(f"pdx-event-reuse: FATAL: {TARGET} not found", file=sys.stderr)
        return 1
    src = TARGET.read_text()
    if "_pdx_reused_event" in src:
        print("    pdx-event-reuse: already applied")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(
            f"pdx-event-reuse: FATAL: expected the save-fence block exactly once, "
            f"found {n}. The image has drifted; re-read worker.py before running "
            f"this arm.",
            file=sys.stderr,
        )
        return 1
    out = src.replace(OLD, NEW, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"pdx-event-reuse: FATAL: patched file does not parse: {e}", file=sys.stderr)
        return 1
    TARGET.write_text(out)
    print("    pdx-event-reuse: applied to worker.py (one event per rank)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
