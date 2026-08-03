#!/usr/bin/env python3
"""Skip the V2 decode warmup when speculative decoding runs under PP.

The two stages deadlock there — one ends up in the sampled-token broadcast while
the other waits on the inter-stage recv — and every rank sits until NCCL times
out (pipelines 60802974 / 60806250 / 60883536). Warmup only pre-JITs kernels, so
skipping decode steps costs a slower first decode rather than correctness.

Delivered by script rather than by context diff: the guard sits in a nested
function whose surrounding lines drift between vLLM builds, and the diff hunk
failed that way in pipeline 60893881. `def _run_decode_step(` is a stable anchor.

Idempotent. Exits non-zero on an unexpected file rather than half-editing it.
"""

import sys

ANCHOR = "        def _run_decode_step(indices: list[int], spec_flags: list[bool]) -> None:"
GUARD = """            if num_spec_steps > 0 and get_pp_group().world_size > 1:
                # Deadlocks: one stage lands in the sampled-token broadcast while
                # the other waits on the inter-stage recv. Warmup only pre-JITs
                # kernels, so skipping is a slower first decode, not a wrong one.
                return
"""
IMPORT = "from vllm.distributed.parallel_state import get_pp_group\n"
MARKER = "num_spec_steps > 0 and get_pp_group().world_size > 1"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: k3_warmup_pp_skip.py <path to vllm/v1/worker/gpu/warmup.py>")
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print("warmup.py already skips decode warmup under PP + spec")
        return

    lines = src.splitlines(keepends=True)

    idx = next((i for i, line in enumerate(lines) if line.rstrip() == ANCHOR), None)
    if idx is None:
        sys.exit(f"ERROR: anchor not found: {ANCHOR.strip()}")

    # Step past the def line and its docstring so the guard is the first statement.
    insert_at = idx + 1
    if insert_at < len(lines) and lines[insert_at].lstrip().startswith('"""'):
        insert_at += 1

    lines = lines[:insert_at] + [GUARD] + lines[insert_at:]

    if IMPORT not in src:
        anchor_import = "from vllm.logger import init_logger\n"
        i = next((k for k, line in enumerate(lines) if line == anchor_import), None)
        if i is None:
            sys.exit("ERROR: could not find the logger import to anchor on")
        lines = lines[:i] + [IMPORT] + lines[i:]

    out = "".join(lines)
    if MARKER not in out:
        sys.exit("ERROR: guard missing after edit")
    compile(out, path, "exec")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("warmup.py patched: decode warmup skipped under PP + spec")


if __name__ == "__main__":
    main()
