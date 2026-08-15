#!/usr/bin/env python3
"""Call truncate_attention_blocks_for_external_load, which nothing was calling.

    python3 configs/patches/apply-privatize-external-pages-callsite.py <site-packages>

Wei's 07839fb50f is in two halves: the method on KVCacheManager, and the nine
lines in Scheduler.schedule that invoke it. The v3 port took the first half and
not the second, because it was cut per-file against the tree the arms run and
`sched/scheduler.py` came back "previously applied" -- the v2 snapshot already
had most of that file, so the whole file was skipped and the one new hunk went
with it.

The result compiled, imported, passed every assertion the setup script makes,
and died at 122/531 requests on exactly the assertion the commit exists to
prevent -- one request off the 123/531 the unpatched v2 stack managed. A method
in the tree is not a method that runs.

The guard against repeating this is at the bottom: assert the call site, not
just the definition.
"""

import pathlib
import sys

TARGET = "vllm/v1/core/sched/scheduler.py"
METHOD = "truncate_attention_blocks_for_external_load"

ANCHOR = """                        connector_prefix_cache_queries = (
"""

ADD = """                        if num_external_computed_tokens > 0:
                            new_computed_blocks = (
                                self.kv_cache_manager
                                .truncate_attention_blocks_for_external_load(
                                    new_computed_blocks,
                                    num_new_local_computed_tokens,
                                )
                            )

"""


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[privatize-callsite] FATAL: {path} not found", file=sys.stderr)
        return 1

    src = path.read_text()
    if METHOD in src:
        print("[privatize-callsite] already present")
        return 0

    # The method has to exist before anything calls it.
    km = site / "vllm/v1/core/kv_cache_manager.py"
    if METHOD not in km.read_text():
        print(f"[privatize-callsite] FATAL: KVCacheManager has no {METHOD}; "
              f"the v3 delta did not apply", file=sys.stderr)
        return 1

    if src.count(ANCHOR) != 1:
        print(f"[privatize-callsite] FATAL: expected one "
              f"connector_prefix_cache_queries assignment, found "
              f"{src.count(ANCHOR)}", file=sys.stderr)
        return 1

    # Both names the inserted block reads must already be bound at that point.
    head = src.split(ANCHOR)[0]
    for name in ("num_external_computed_tokens", "new_computed_blocks",
                 "num_new_local_computed_tokens"):
        if name not in head:
            print(f"[privatize-callsite] FATAL: {name} is not in scope above "
                  f"the insertion point", file=sys.stderr)
            return 1

    path.write_text(src.replace(ANCHOR, ADD + ANCHOR, 1))
    print(f"[privatize-callsite] applied: scheduler now calls {METHOD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
