#!/usr/bin/env python3
"""Count where MooncakeStoreWorker.lookup() leaves, and why.

    python3 configs/patches/apply-lookup-exit-log.py <site-packages>

The connector's own counters, from a c48 arm serving ~3,000 requests in the
window:

    lookup_exists_count:  8
    save_exists_count:  240
    save_put_count:     176        save_put_total_bytes: 23,611,834,368

Eight lookups against three thousand requests. Whatever is happening, most
requests are not reaching the store at all -- and `lookup_exists_error_count`
is 0, which proves nothing: an exception raised before the counter increments
is invisible to it. That is exactly how a NameError in this same connector hid
for a day while the client sat at "0 returned, 0 errors".

`lookup()` has three early exits before it ever asks the store:

    if self._capacity_only:                     return 0
    token_len = align_lookup_length(num_tokens)
    if not block_hashes or token_len <= 0:      return 0

and then the body, which can raise.

This instruments all of them and prints a tally every 128 calls, so the answer
arrives as a distribution rather than an anecdote:

    [mooncake-lookup] calls=N capacity_only=A no_hashes=B zero_len=C
                      asked=D raised=E last_err=...

If `asked` is near zero the store is never queried and the key-derivation work
is beside the point. If `raised` is non-zero we have been reading a silent
exception as an empty cache all along.
"""

import pathlib
import sys

TARGET = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"

BEFORE = '''    def lookup(self, num_tokens: int, block_hashes: Sequence[BlockHash]) -> int:
'''

# Wrap rather than edit the body: the body is long, Wei's branch rewrites parts
# of it, and a wrapper cannot drift with the parts it does not name.
AFTER = '''    def _lookup_instrumented(
        self, num_tokens: int, block_hashes: Sequence[BlockHash]
    ) -> int:
        c = getattr(self, "_lk_counts", None)
        if c is None:
            c = self._lk_counts = {
                "calls": 0, "capacity_only": 0, "no_hashes": 0,
                "zero_len": 0, "asked": 0, "raised": 0, "hit": 0,
            }
        c["calls"] += 1
        if self._capacity_only:
            c["capacity_only"] += 1
        elif not block_hashes:
            c["no_hashes"] += 1
        elif self.coord.align_lookup_length(num_tokens) <= 0:
            c["zero_len"] += 1
        else:
            c["asked"] += 1
        try:
            out = self._lookup_orig(num_tokens, block_hashes)
        except BaseException as e:
            c["raised"] += 1
            self._lk_last_err = f"{type(e).__name__}: {e}"
            if c["raised"] <= 3:
                logger.exception("[mooncake-lookup] lookup raised")
            raise
        if out:
            c["hit"] += 1
        if c["calls"] % 128 == 0:
            logger.info(
                "[mooncake-lookup] calls=%d capacity_only=%d no_hashes=%d "
                "zero_len=%d asked=%d raised=%d returned_nonzero=%d last_err=%s",
                c["calls"], c["capacity_only"], c["no_hashes"], c["zero_len"],
                c["asked"], c["raised"], c["hit"],
                getattr(self, "_lk_last_err", "<none>"),
            )
        return out

    def _lookup_orig(self, num_tokens: int, block_hashes: Sequence[BlockHash]) -> int:
'''


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[lookup-exit] FATAL: {path} not found", file=sys.stderr)
        return 1

    src = path.read_text()
    if "_lookup_instrumented" in src:
        print("[lookup-exit] already present")
        return 0
    if src.count(BEFORE) != 1:
        print(f"[lookup-exit] FATAL: {src.count(BEFORE)} definitions of "
              f"lookup(), expected 1", file=sys.stderr)
        return 1

    src = src.replace(BEFORE, AFTER, 1)
    # Rebind at the end of the class so callers reach the wrapper. Appending a
    # module-level assignment is safer than editing the class body, which Wei's
    # branch rewrites.
    cls = "class MooncakeStoreWorker"
    if cls not in src:
        print(f"[lookup-exit] FATAL: no {cls}", file=sys.stderr)
        return 1
    src += (
        "\n\n# Route lookup() through the instrumented wrapper. Assigned after the\n"
        "# class body so the wrapper can call the original by its new name.\n"
        "MooncakeStoreWorker.lookup = MooncakeStoreWorker._lookup_instrumented\n"
    )
    path.write_text(src)
    print("[lookup-exit] applied: lookup() exits are counted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
