#!/usr/bin/env python3
"""Log why an external hit does or does not turn into a load.

    python3 configs/patches/apply-external-hit-decision-log.py <site-packages>

`External prefix cache hit rate: 0.0%` has been read all day as "the store has
nothing". The scheduler says something narrower:

    if num_external_hit_tokens < num_computed_tokens:
        need_to_allocate = 0
    ...
    if need_to_allocate <= 0:
        return 0, False

The store only contributes what the GPU prefix cache does not already hold. Our
local hit rate is 87-94%, so a perfectly healthy store that returns *less* than
the local cache reports exactly the same 0.0%, never calls load, and never
records a `load_get`. That is consistent with every counter observed:
lookup_exists runs over 213,472 keys, save_put writes 22 GB with zero failures,
and the load path is silent.

Which of the two it is -- nothing found, or found-but-beaten -- is decided by a
single number that vLLM logs at DEBUG, a level these arms do not run at. So
this raises exactly that decision to INFO, rate-limited, and prints both sides
of the comparison rather than the verdict alone:

    [mooncake-decision] external=N local=M -> load M2 tokens   (or: no load)

If external is 0, the lookup genuinely finds nothing and the key derivation is
the place to look. If external is large but below local, the tier works and
this workload simply has little for it to add -- and the offload hypothesis for
the c64 divergence dies on the spot.
"""

import pathlib
import sys

TARGET = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py"

BEFORE = """        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens
"""

AFTER = '''        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # Rate-limited to one line per 256 decisions: this runs per request and
        # the point is the distribution, not every instance.
        _n = getattr(self, "_mooncake_decision_n", 0)
        self._mooncake_decision_n = _n + 1
        if _n % 256 == 0:
            logger.info(
                "[mooncake-decision] tokens=%d external=%d local=%d -> %s",
                request.num_tokens,
                num_external_hit_tokens,
                num_computed_tokens,
                (f"load {need_to_allocate}" if need_to_allocate > 0
                 else "no load (external does not beat the local cache)"),
            )
'''


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[hit-decision] FATAL: {path} not found", file=sys.stderr)
        return 1

    src = path.read_text()
    if "_mooncake_decision_n" in src:
        print("[hit-decision] already present")
        return 0
    if src.count(BEFORE) != 1:
        print(f"[hit-decision] FATAL: {src.count(BEFORE)} matches for the "
              f"need_to_allocate branch, expected 1", file=sys.stderr)
        return 1
    # It logs, so the module must have a logger. This file does; assert rather
    # than assume, because assuming it cost four ladders once already.
    if "logger" not in src.split("\n\n")[0] and "init_logger" not in src:
        print("[hit-decision] FATAL: scheduler.py defines no logger",
              file=sys.stderr)
        return 1

    path.write_text(src.replace(BEFORE, AFTER, 1))
    print("[hit-decision] applied: the external-vs-local decision is logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
