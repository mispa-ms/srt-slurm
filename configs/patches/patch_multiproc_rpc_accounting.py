#!/usr/bin/env python3
"""Count commands sent against responses written, on both sides of the executor.

Arm J settled what the wedge is not. Both message queues are in a legitimate
"everything consumed, waiting for the next message" state -- written=1 with every
read flag set -- the ring indices are not desynchronised, nobody is blocked in
acquire_write, and no worker raised. So the engine is waiting for a response that
was never produced, and FutureWrapper.result() drains the futures queue until one
of them blocks forever:

    while not self.done():
        future = self.futures_queue.pop()
        future._wait_for_response()

What that does not say is where the mismatch enters. Four counters answer it
directly, and they are cheap: increment in memory, print only on the wait warning
that already fires every 60 s.

    engine   sent  commands enqueued on rpc_broadcast_mq
             recv  responses consumed by _wait_for_response
    worker   cmds  commands dequeued in worker_busy_loop
             outs  outputs written by handle_output

At the wedge one of two shapes appears, and they point at different fixes:

    sent == recv + 1, worker cmds == outs
        the engine registered a future for a command that was never sent, or one
        response was consumed by the wrong future. The bug is engine-side, in
        collective_rpc or the futures queue.

    sent == recv + 1, worker cmds == outs + 1
        the worker took a command and never answered it. The bug is worker-side,
        in worker_busy_loop or handle_output.

The last few (method, output_rank) pairs come along too, so the outstanding RPC
is named rather than inferred from whichever one happened to time out.

All of it is wrapped in try/except -- instrumentation must not be able to change
the outcome it is measuring.
"""

import ast
import pathlib
import sys

import vllm.distributed.device_communicators.shm_broadcast as shm
import vllm.v1.executor.multiproc_executor as mpe


def patch(path: pathlib.Path, edits: list[tuple[str, str, str]]) -> None:
    src = path.read_text()
    for name, old, new in edits:
        if new in src:
            print(f"[rpcdbg] {name} already patched")
            continue
        if old not in src:
            print(f"[rpcdbg] FATAL - {name} anchor not found in {path}", file=sys.stderr)
            raise SystemExit(1)
        src = src.replace(old, new, 1)
        print(f"[rpcdbg] patched {name}")
    ast.parse(src)
    path.write_text(src)


# --- shm_broadcast: hold the counters and print them on the wait warning -------
shm_path = pathlib.Path(shm.__file__)
patch(
    shm_path,
    [
        (
            "counter store",
            "SHM_READER_RECHECK_INTERVAL_MS = 5000",
            "SHM_READER_RECHECK_INTERVAL_MS = 5000\n\n"
            "# Filled in by multiproc_executor; printed on the long-wait warning.\n"
            "RPC_DBG: dict = {}",
        ),
        (
            "reader warning",
            '                            "[shmdbg] READER blocked shm=%s rank=%d idx=%d/%d "\n'
            '                            "written=%d my_read_flag=%d all_flags=%s",',
            '                            "[shmdbg] READER blocked shm=%s rank=%d idx=%d/%d "\n'
            '                            "written=%d my_read_flag=%d all_flags=%s rpc=%s",',
        ),
        (
            "reader warning args",
            "                            metadata_buffer[self.local_reader_rank + 1],\n"
            "                            list(metadata_buffer[0:]),\n"
            "                        )",
            "                            metadata_buffer[self.local_reader_rank + 1],\n"
            "                            list(metadata_buffer[0:]),\n"
            "                            RPC_DBG,\n"
            "                        )",
        ),
    ],
)

# --- multiproc_executor: keep the counts --------------------------------------
mpe_path = pathlib.Path(mpe.__file__)
patch(
    mpe_path,
    [
        (
            "import",
            "logger = init_logger(__name__)",
            "logger = init_logger(__name__)\n\n"
            "from vllm.distributed.device_communicators import shm_broadcast as _rpcdbg_shm\n\n\n"
            "def _rpcdbg(key, value=None):\n"
            "    try:\n"
            "        d = _rpcdbg_shm.RPC_DBG\n"
            "        if value is None:\n"
            "            d[key] = d.get(key, 0) + 1\n"
            "        else:\n"
            "            d[key] = value\n"
            "    except Exception:\n"
            "        pass",
        ),
        (
            "engine sent",
            "        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))",
            "        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))\n"
            "        _rpcdbg('sent')\n"
            "        _rpcdbg('last_sent', f'{method if isinstance(method, str) else \"<fn>\"}"
            "/rank{output_rank}')",
        ),
        (
            "engine recv",
            "            response = self.aggregate(self.get_response())\n"
            "            with suppress(InvalidStateError):",
            "            response = self.aggregate(self.get_response())\n"
            "            _rpcdbg('recv')\n"
            "            with suppress(InvalidStateError):",
        ),
        (
            "worker cmds",
            "                if isinstance(method, str):\n"
            "                    func = getattr(self.worker, method)",
            "                _rpcdbg('cmds')\n"
            "                _rpcdbg('last_cmd', f'{method if isinstance(method, str) else \"<fn>\"}"
            "/rank{output_rank}')\n"
            "                if isinstance(method, str):\n"
            "                    func = getattr(self.worker, method)",
        ),
        (
            "worker outs",
            "                if output_rank is None or self.rank == output_rank:\n"
            "                    self.handle_output(output)",
            "                if output_rank is None or self.rank == output_rank:\n"
            "                    self.handle_output(output)\n"
            "                    _rpcdbg('outs')",
        ),
    ],
)

print("[rpcdbg] done")
