#!/usr/bin/env python3
"""Keep the Mooncake store's per-step CUDA event alive on the main thread.

WHY. At every aws-pdx stall the store's KVCacheStoreSendingThread is stopped at
`pthread_rwlock_wrlock` inside `cuEventDestroy_v2`, reached from
`THCPEvent_dealloc`, while the worker's main thread spins in `sched_yield` under
`cuGraphLaunch`. Verbatim, from the idle rank of run 427249:

    KVCacheStoreSendingThread        MainThread
      pthread_rwlock_wrlock            sched_yield
      cuEventDestroy_v2                cuGraphLaunch
      cudaEventDestroy                 cudaGraphLaunch
      THCPEvent_dealloc                replay (breakable_cudagraph.py:214)
      run (store/worker.py:614)        run_pw_graph (cudagraph_utils.py:460)

The connector allocates one `torch.cuda.Event` per step on the main thread
(`worker.py:2159`), hands it to the sending thread, and the sending thread drops
the last reference at the top of its loop (`worker.py:614`, `request_data =
None`). So a `cudaEventDestroy` runs on a non-main thread once per saved step,
concurrent with the main thread's graph launches. That also explains why the
stall needs the save path -- no save, no event, no cross-thread destroy.

WHAT THIS DOES. The main thread appends each event to a bounded ring it owns, so
the sending thread's drop is never the last reference. Eviction -- and therefore
`cudaEventDestroy` -- happens on the main thread, which is the thread that owns
the launches, hundreds of steps after the event completed.

This changes no semantics: the event is still recorded on the compute stream and
still synchronized by the sending thread. It only moves who frees it.

WHAT IT IS NOT. It is not a fix until an A/B says so. It is the cheapest test of
the hypothesis that does not need a Mooncake rebuild, since the copy itself lives
in store.so and cannot be touched from Python.
"""

import ast
import pathlib
import sys

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/"
    "kv_connector/v1/mooncake/store/worker.py"
)

# The exact block, as it is in image 3696c772a. Matching the whole block rather
# than one line means a drifted image fails loudly here instead of silently
# applying nothing.
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
                    current_event = torch.cuda.Event()
                    current_event.record()
                    # PDX: keep the event alive on this thread. Dropping the last
                    # reference on the sending thread runs cudaEventDestroy there,
                    # which blocks on a libcuda writer lock while this thread is
                    # inside cuGraphLaunch. Eviction frees it here instead.
                    _ring = getattr(self, "_pdx_event_ring", None)
                    if _ring is None:
                        import collections as _pdx_collections
                        _ring = self._pdx_event_ring = _pdx_collections.deque(
                            maxlen=int(__import__("os").environ.get(
                                "PDX_STORE_EVENT_RING", "256"))
                        )
                    _ring.append(current_event)
                    break
"""


def main() -> int:
    if not TARGET.exists():
        print(f"pdx-event-pool: FATAL: {TARGET} not found", file=sys.stderr)
        return 1
    src = TARGET.read_text()
    if "_pdx_event_ring" in src:
        print("    pdx-event-pool: already applied")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(
            f"pdx-event-pool: FATAL: expected the save-fence block exactly once, "
            f"found {n}. The image has drifted; re-read worker.py before running "
            f"this arm.",
            file=sys.stderr,
        )
        return 1
    out = src.replace(OLD, NEW, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"pdx-event-pool: FATAL: patched file does not parse: {e}", file=sys.stderr)
        return 1
    TARGET.write_text(out)
    print(
        f"    pdx-event-pool: applied to worker.py "
        f"(ring size {__import__('os').environ.get('PDX_STORE_EVENT_RING', '256')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
