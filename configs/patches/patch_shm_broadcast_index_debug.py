#!/usr/bin/env python3
"""Log ring-buffer indices next to the shm_broadcast long-wait warning.

Twelve hypotheses are dead and the deadlock still looks the same every time: after
an nsys capture, EngineCore sits in acquire_read waiting for a response while all
eight workers sit in acquire_read waiting for a command. Neither side is in
acquire_write, so the ring is not full, and both sides re-check their flags every
5 s -- a lost notification cannot explain it.

What is left is that writer and readers are looking at different slots. That would
produce exactly this: the writer's slot is free so it never warns, the readers'
slot is unwritten so they wait, and the writer does not advance because it is
itself waiting for a reply.

`current_idx` is per-process local state, so py-spy cannot see it. But both wait
paths already log every VLLM_RINGBUFFER_WARNING_INTERVAL seconds -- 30-odd times
across a wedge -- and that message is the natural place to carry the state.

Also splits the message per role. Upstream uses one string for both acquire_write
and acquire_read, which is why an earlier grep over these logs could not tell
which side was stuck and the py-spy stacks had to settle it.
"""

import pathlib
import sys

import vllm.distributed.device_communicators.shm_broadcast as shm

path = pathlib.Path(shm.__file__)
src = path.read_text()

WRITER_OLD = """                    if elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                        logger.info(
                            LONG_WAIT_TIME_LOG_MSG, VLLM_RINGBUFFER_WARNING_INTERVAL
                        )
                        n_warning += 1"""

WRITER_NEW = """                    if elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                        logger.info(
                            "[shmdbg] WRITER blocked %.0fs shm=%s idx=%d/%d "
                            "written=%d read_flags=%s n_reader=%d",
                            elapsed,
                            getattr(self.buffer.shared_memory, "name", "?"),
                            self.current_idx,
                            self.buffer.max_chunks,
                            metadata_buffer[0],
                            list(metadata_buffer[1:]),
                            self.buffer.n_reader,
                        )
                        n_warning += 1"""

READER_OLD = """                    # if we wait for a long time, log a message
                    if read_timeout.should_warn():
                        logger.info(
                            LONG_WAIT_TIME_LOG_MSG, VLLM_RINGBUFFER_WARNING_INTERVAL
                        )"""

READER_NEW = """                    # if we wait for a long time, log a message
                    if read_timeout.should_warn():
                        logger.info(
                            "[shmdbg] READER blocked shm=%s rank=%d idx=%d/%d "
                            "written=%d my_read_flag=%d all_flags=%s",
                            getattr(self.buffer.shared_memory, "name", "?"),
                            self.local_reader_rank,
                            self.current_idx,
                            self.buffer.max_chunks,
                            metadata_buffer[0],
                            metadata_buffer[self.local_reader_rank + 1],
                            list(metadata_buffer[0:]),
                        )"""

for name, old, new in (
    ("writer", WRITER_OLD, WRITER_NEW),
    ("reader", READER_OLD, READER_NEW),
):
    if new in src:
        print(f"[shmdbg] {name} already patched")
        continue
    if old not in src:
        print(f"[shmdbg] FATAL - {name} anchor not found in {path}", file=sys.stderr)
        raise SystemExit(1)
    src = src.replace(old, new, 1)
    print(f"[shmdbg] patched {name}")

# The readers wait indefinitely and so never warn at all -- should_warn is gated on
# `not indefinite`. That is the half of the picture we most need, so make the
# workers talk too.
IND_OLD = """        read_timeout = self.ReadTimeoutWithWarnings(
            timeout=timeout, should_warn=not indefinite
        )"""
IND_NEW = """        read_timeout = self.ReadTimeoutWithWarnings(timeout=timeout, should_warn=True)"""
if IND_NEW in src:
    print("[shmdbg] indefinite-warn already patched")
elif IND_OLD in src:
    src = src.replace(IND_OLD, IND_NEW, 1)
    print("[shmdbg] patched indefinite readers to warn")
else:
    print("[shmdbg] FATAL - indefinite anchor not found", file=sys.stderr)
    raise SystemExit(1)

path.write_text(src)

import ast

ast.parse(src)
print(f"[shmdbg] done, {path} parses clean")
