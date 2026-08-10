#!/usr/bin/env python3
"""Poll the async-output copy event instead of blocking-waiting on it.

The wedge, finally. After an nsys capture the worker's WorkerAsyncOutputCopy
thread parks here and never comes back:

    synchronize (torch/cuda/streams.py:254)
    get_output (vllm/v1/worker/gpu/async_utils.py:69)
    enqueue_output (vllm/v1/executor/multiproc_executor.py:984)
    async_output_busy_loop (vllm/v1/executor/multiproc_executor.py:1022)

Everything else follows. handle_output only puts on an in-process queue, so the
worker's own counters keep advancing (cmds 96416, outs 96416) while nothing
reaches the response queue and the engine stays three replies short (sent 96416,
recv 96413). The engine then blocks in FutureWrapper.result() until
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS, and the RPC it names -- sample_tokens -- is
just whichever one was outstanding. Which is why turning DSpark off, dropping GPU
metrics, shrinking the window and disabling async scheduling all changed nothing.

The event is created with

    # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
    self.copy_event = torch.cuda.Event(blocking=True)

blocking=True is cudaEventBlockingSync: the thread sleeps on a driver primitive
instead of spinning. That is a throughput choice, as the comment says, not a
correctness requirement -- and it is the path that does not survive CUPTI
teardown. cudaEventQuery does not use it.

So: spin briefly on query(), then poll with a short sleep, and give up loudly
after a deadline rather than hang until something else times out. If the event is
in fact complete and only the blocking wait was broken, the first query() returns
immediately and the hang is gone. If the copy genuinely never lands, the run dies
in seconds saying so.

Both call sites are patched -- the sampler path and the pooler path -- because
the container's line numbers do not match any one checkout and only one of them
appeared in the stack.
"""

import ast
import pathlib
import sys

import vllm.v1.worker.gpu.async_utils as au

path = pathlib.Path(au.__file__)
src = path.read_text()

HELPER = '''

def _wait_copy_event(event) -> None:
    """Wait for a copy event without cudaEventBlockingSync.

    Spin first, because the copy is normally already done by the time anyone
    asks; then sleep, so eight workers do not each hold a core; then raise, so a
    wedged copy stream is a fast loud failure instead of a silent hang that only
    surfaces as an unrelated RPC timeout half an hour later.
    """
    timeout = float(os.environ.get("VLLM_ASYNC_OUTPUT_COPY_TIMEOUT_S", "120"))
    start = time.monotonic()
    while not event.query():
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise RuntimeError(
                f"async output copy event did not complete in {timeout}s -- "
                "the copy stream is wedged (seen after nsys capture teardown)"
            )
        if elapsed > 0.005:
            time.sleep(0.0005)
'''

# ---- 1. imports --------------------------------------------------------------
if "import time" not in src.split("class ")[0]:
    anchor = "import contextlib\n"
    if anchor not in src:
        print("[asyncevt] FATAL - import anchor not found", file=sys.stderr)
        raise SystemExit(1)
    src = src.replace(anchor, "import contextlib\nimport os\nimport time\n", 1)
    print("[asyncevt] patched imports")
else:
    print("[asyncevt] imports already patched")

# ---- 2. helper ---------------------------------------------------------------
if "_wait_copy_event" in src:
    print("[asyncevt] helper already present")
else:
    marker = "\nclass AsyncOutput("
    if marker not in src:
        print("[asyncevt] FATAL - class anchor not found", file=sys.stderr)
        raise SystemExit(1)
    src = src.replace(marker, HELPER + marker, 1)
    print("[asyncevt] inserted helper")

# ---- 3. every blocking event -> non-blocking ---------------------------------
n = src.count("torch.cuda.Event(blocking=True)")
if n:
    src = src.replace(
        "        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.\n"
        "        self.copy_event = torch.cuda.Event(blocking=True)",
        "        # Was a blocking (sleep) event -- cudaEventBlockingSync waits on a\n"
        "        # driver primitive and does not wake again after an nsys capture ends.\n"
        "        self.copy_event = torch.cuda.Event(blocking=False)",
    )
    src = src.replace("torch.cuda.Event(blocking=True)", "torch.cuda.Event(blocking=False)")
    print(f"[asyncevt] switched {n} event(s) to non-blocking")
else:
    print("[asyncevt] events already non-blocking")

# ---- 4. every synchronize -> polled wait -------------------------------------
n = src.count("self.copy_event.synchronize()")
if n:
    src = src.replace("self.copy_event.synchronize()", "_wait_copy_event(self.copy_event)")
    print(f"[asyncevt] replaced {n} synchronize call(s)")
else:
    print("[asyncevt] synchronize calls already replaced")

if "self.copy_event.synchronize()" in src or "Event(blocking=True)" in src:
    print("[asyncevt] FATAL - a call site survived the rewrite", file=sys.stderr)
    raise SystemExit(1)

ast.parse(src)
path.write_text(src)
print(f"[asyncevt] done, {path} parses clean")
