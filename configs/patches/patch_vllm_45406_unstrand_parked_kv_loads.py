"""Backport vLLM PR #45406: don't strand parked async-KV-load requests.

Upstream: https://github.com/vllm-project/vllm/pull/45406
  "[BugFix] Don't strand parked async-KV-load requests behind an unschedulable
   queue head" -- open since 2026-06-12, never merged.

Symptom, from issue #45387 and reproduced on every one of our conc>=16 disagg
runs: the scheduler wedges with Running=0, Waiting=N, Deferred=N, GPU KV at or
near 0%, errors=0, and stays there until the walltime kills the job. No
assertion, no engine death, no memory pressure -- it simply never schedules
again.

Mechanism. Requests parked in WAITING_FOR_REMOTE_KVS hold their allocated
blocks and are only promoted when the waiting-queue traversal reaches them. When
the head of that queue cannot be allocated, the traversal `break`s. If nothing
is running, no future event can free blocks: the head keeps failing
allocate_slots, the parked requests behind it are never promoted, and schedule()
returns an empty step forever.

The fix keeps the `break` whenever something is running -- that preserves
queue-order admission, since running requests will free blocks when they
finish -- and only scans past an unschedulable head when the engine is
otherwise idle, which is exactly the state that cannot recover on its own.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

OLD = """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break
"""

NEW = """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if self.running:
                        # Running requests will free blocks when they
                        # complete; stop here to preserve queue-order
                        # admission.
                        break
                    # Nothing is running, so no future event frees blocks and
                    # stopping at this request would freeze this state
                    # permanently. Requests behind this one may hold blocks
                    # while parked (async KV loads in WAITING_FOR_REMOTE_KVS)
                    # and are only promoted when this traversal reaches them.
                    # Keep scanning so they can be promoted, scheduled, and
                    # eventually free the blocks this request needs.
                    # See https://github.com/vllm-project/vllm/issues/45388
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue
"""


def patch_scheduler(package_dir: Path) -> None:
    target = package_dir / "v1/core/sched/scheduler.py"
    source = target.read_text()
    if "Nothing is running, so no future event frees blocks" in source:
        print(f"[vllm-45406] Already patched {target}")
        return
    if source.count(OLD) != 1:
        raise RuntimeError(
            f"Expected exactly one anchor in {target}, found {source.count(OLD)}"
        )
    source = source.replace(OLD, NEW, 1)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"[vllm-45406] Patched {target}")


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed vllm package")
    patch_scheduler(Path(next(iter(spec.submodule_search_locations))))


if __name__ == "__main__":
    main()
