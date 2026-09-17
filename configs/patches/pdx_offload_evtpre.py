#!/usr/bin/env python3
"""Pre-allocate OffloadingConnector's CUDA events instead of making them mid-flight.

WHAT SPLITS THE THREE CONNECTORS. Two hang, one does not, on the same workload.
It is not the copy mechanism -- `_select_swap_blocks_fn` returns the C++ batch
DMA (`ops.swap_blocks_batch`) for every save, and the Triton path is only
reachable on load below `THRESHOLD_BYTES = 28 KiB` while a K3 page is 884,736 B.
It is not stream priority -- `priority_range()[0]` is 0, the default. What is
left is when their CUDA events are created:

    Mooncake             hangs   created and destroyed per step from the store
                                 worker thread (store/worker.py:2159 create and
                                 record, :614 drops the last ref into
                                 THCPEvent_dealloc)
    OffloadingConnector  hangs   two per transfer, made on demand whenever the
                                 pool is empty, as torch.Event(enable_timing=True)
    SimpleCPUOffload     clean   pre-allocated on the main thread, no timing

Both hangers create and release events from the transfer path while the model is
running; the survivor does not. That also reconnects to the oldest observation
here: every early stall dump had the store's sending thread stopped in
`cuEventDestroy_v2 -> __pthread_rwlock_wrlock`, which was demoted at 3 of 44 as
an explanation for "the copy is stuck" but was never tested as event lifetime
with any statistical power.

WHAT THIS CHANGES, AND NOTHING ELSE:

  1. fill `_event_pool` at construction, so `transfer_async` never has to make
     one while transfers are in flight;
  2. drop `enable_timing=True` from the fallback path, matching the survivor.

The pool is a plain list used as a stack and returned to on completion, so
seeding it is behaviour-preserving apart from removing the allocation.

Off unless PDX_OFFLOAD_EVTPRE=1, so the control arm is byte-identical.
Diagnostics only: any problem is reported and the file is left untouched.
"""

import os
import re
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/gpu_worker.py"
MARKER = "# pdx-offload-evtpre"
POOL_SIZE = int(os.environ.get("PDX_OFFLOAD_EVTPRE_SIZE", "64"))


def main() -> int:
    if os.environ.get("PDX_OFFLOAD_EVTPRE", "").strip() not in ("1", "true", "yes"):
        print("    offload-evtpre: off (set PDX_OFFLOAD_EVTPRE=1 to enable)")
        return 0

    path = os.environ.get("PDX_OFFLOAD_EVTPRE_TARGET", TARGET)
    try:
        src = open(path).read()
    except OSError as e:
        print(f"    offload-evtpre: WARNING cannot read {path}: {e}")
        return 0
    if MARKER in src:
        print("    offload-evtpre: already applied")
        return 0

    # 1. Seed the pool where it is created.
    old_init = "        self._event_pool: list[torch.Event] = []\n"
    if old_init not in src:
        print("    offload-evtpre: WARNING the event pool is not in the expected "
              "shape; leaving the file untouched")
        return 0
    new_init = (
        "        self._event_pool: list[torch.Event] = [\n"
        f"            torch.Event() for _ in range({POOL_SIZE})\n"
        "        ]  " + MARKER + "\n"
    )
    out = src.replace(old_init, new_init, 1)

    # 2. Match the survivor on the fallback path too.
    n = out.count("else torch.Event(enable_timing=True)")
    if n == 0:
        print("    offload-evtpre: WARNING no on-demand event creation found; "
              "leaving the file untouched")
        return 0
    out = out.replace("else torch.Event(enable_timing=True)",
                      "else torch.Event()  " + MARKER)

    if not re.search(r"^import torch$", out, re.M):
        print("    offload-evtpre: WARNING torch is not imported at module scope")
        return 0
    try:
        open(path, "w").write(out)
    except OSError as e:
        print(f"    offload-evtpre: WARNING cannot write {path}: {e}")
        return 0
    print(f"    offload-evtpre: applied (pool={POOL_SIZE}, {n} on-demand sites) "
          f"to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
