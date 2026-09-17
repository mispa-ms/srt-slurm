#!/usr/bin/env python3
"""Give OffloadingConnector's transfer streams the lowest priority.

WHAT THIS TESTS. Three KV-offload connectors, two of which hang:

    Mooncake             hangs   synchronous cudaMemcpy D2H on the LEGACY
                                 default stream (no _ptsz symbols in store.so
                                 or engine.so) plus GPU-sourced RDMA
    OffloadingConnector  hangs   a Triton kernel, _swap_blocks_kernel, so the
                                 copy occupies SMs and the launch queue; its
                                 streams are created with no priority
                                 (gpu_worker.py:363, current_platform.Stream())
    SimpleCPUOffload     clean   cuMemcpyBatchAsync -- a driver batch DMA --
                                 on two dedicated streams created at the
                                 LOWEST priority (priority_range()[0])

The two that hang both put the KV read into the GPU's scheduling path as work
that competes with the model: Mooncake through the legacy default stream, which
implicitly synchronises with every blocking stream in the context, and
OffloadingConnector through actual kernels. The one that stays up uses a DMA
that needs no SMs and a stream that yields.

That gives two separable candidates -- stream priority, and the copy mechanism.
This patch changes ONLY the first, in one line, so a pass or a fail separates
them:

    pass  -> contention/priority is the mechanism, and the same knob is worth
             trying wherever else the copy shares the compute stream
    fail  -> priority is not enough and the mechanism is SM occupancy itself,
             which points at replacing the Triton copy with the batch DMA that
             SimpleCPUOffload already uses

Off unless PDX_OFFLOAD_LOWPRI=1, so the control arm is byte-identical.
Diagnostics only: any problem is reported and the file is left untouched.
"""

import os
import re
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/gpu_worker.py"
MARKER = "# pdx-offload-lowpri"


def main() -> int:
    if os.environ.get("PDX_OFFLOAD_LOWPRI", "").strip() not in ("1", "true", "yes"):
        print("    offload-lowpri: off (set PDX_OFFLOAD_LOWPRI=1 to enable)")
        return 0

    path = os.environ.get("PDX_OFFLOAD_LOWPRI_TARGET", TARGET)
    try:
        src = open(path).read()
    except OSError as e:
        print(f"    offload-lowpri: WARNING cannot read {path}: {e}")
        return 0

    if MARKER in src:
        print("    offload-lowpri: already applied")
        return 0

    # Match the expression, not a line number: the file is the image's, and a
    # line number taken from a different checkout has already misled this
    # investigation once.
    pat = re.compile(
        r"(?P<head>self\._stream_pool\.pop\(\) if self\._stream_pool else )"
        r"current_platform\.Stream\(\)"
    )
    m = pat.search(src)
    if not m:
        print("    offload-lowpri: WARNING the stream allocation is not in the "
              "expected shape; leaving the file untouched")
        return 0

    # Mirror SimpleCPUOffload exactly: priority_range() returns
    # (least_priority, greatest_priority) and it takes the first.
    repl = (m.group("head")
            + "current_platform.Stream("
            + "priority=torch.cuda.Stream.priority_range()[0])  " + MARKER)
    out = src[: m.start()] + repl + src[m.end():]

    if not re.search(r"^import torch$", out, re.M):
        print("    offload-lowpri: WARNING torch is not imported at module "
              "scope; leaving the file untouched")
        return 0

    try:
        open(path, "w").write(out)
    except OSError as e:
        print(f"    offload-lowpri: WARNING cannot write {path}: {e}")
        return 0

    print(f"    offload-lowpri: applied to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
