#!/usr/bin/env python3
"""Log one written key and one queried key, once per process.

The mooncake master's own counters say the store works and is never read:

    Mem Storage: 380.16 GB / 480.00 GB (79.2%) | Keys: 19224 | Clients: 4
    Get=0.00/0.00

19,224 keys live, 380 GB resident, and not a single Get in a 60-minute run,
while the lookup asks about ~240,000 keys per window and reports no error. So
the keys being asked for are not the keys that were written -- but *which* axis
disagrees (the block alignment, the dcp_rank namespace, the metadata prefix) is
still unknown, and every attempt to reason it out from the code has produced a
plausible theory that the next run refuted.

A key is a string. Print one from each side and the answer is in the diff, with
nothing left to infer. Once per process: the question is whether the two forms
agree, and that does not change between requests.

Applied as a string replacement rather than a .patch for the same reason as
apply-lookup-align.py -- two hand-made diffs for a hunk in this file failed on
context and on header format, neither related to the change.
"""

import pathlib
import sys

TARGET = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"

# --- lookup side: the candidate list is complete here -----------------------
LOOKUP_BEFORE = """        if not candidate_keys:
            return 0
"""

LOOKUP_AFTER = '''        if not candidate_keys:
            return 0

        if not getattr(self, "_logged_key_sample_lookup", False):
            self._logged_key_sample_lookup = True
            # The whole prefix SET, not one key. A hash counts as a hit only if
            # every namespace in this list exists (all(res[pos+j] == 1) below),
            # while the save path writes exactly one -- this rank's own, from
            # PoolKey.build_prefix(metadata) with no overrides. Printing a
            # single candidate would invite comparing save's tp_rank against
            # whichever namespace happens to sort first.
            logger.info(
                "[mooncake-keysample] LOOKUP asks %d keys over %d prefixes, "
                "token_len=%d, fine_grained=%s",
                len(candidate_keys),
                len(self._lookup_key_prefixes[0]),
                token_len,
                fine_grained,
            )
            for _p in self._lookup_key_prefixes[0]:
                logger.info("[mooncake-keysample]   LOOKUP prefix: %s", _p)
'''

# --- save side: immediately before the put that fills the store -------------
SAVE_BEFORE = """        batch_bytes = _sum_batch_bytes(sizes)
        put_start = time.perf_counter()
        try:
            res = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, self.replicate_config
            )
"""

SAVE_AFTER = '''        batch_bytes = _sum_batch_bytes(sizes)
        if keys and not getattr(self, "_logged_key_sample_save", False):
            self._logged_key_sample_save = True
            logger.info(
                "[mooncake-keysample] SAVE puts %d keys with prefix: %s",
                len(keys),
                keys[0].rsplit("@", 1)[0],
            )
        put_start = time.perf_counter()
        try:
            res = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, self.replicate_config
            )
'''


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[keysample] FATAL: {path} not found", file=sys.stderr)
        return 1

    src = path.read_text()
    if "_logged_key_sample_lookup" in src:
        print("[keysample] already present")
        return 0

    # The second put site is indented one level deeper and wraps its arguments
    # one per line, so the first anchor cannot match it however many
    # replacements are allowed. It is the site the arms actually take.
    SAVE2_BEFORE = """            batch_bytes = _sum_batch_bytes(sizes)
            put_start = time.perf_counter()
"""
    SAVE2_AFTER = '''            batch_bytes = _sum_batch_bytes(sizes)
            if keys and not getattr(self, "_logged_key_sample_save", False):
                self._logged_key_sample_save = True
                logger.info(
                    "[mooncake-keysample] SAVE puts %d keys with prefix: %s",
                    len(keys),
                    keys[0].rsplit("@", 1)[0],
                )
            put_start = time.perf_counter()
'''

    for what, before, after, only_first in (
        ("lookup", LOOKUP_BEFORE, LOOKUP_AFTER, True),
        ("save-deep", SAVE2_BEFORE, SAVE2_AFTER, False),
        # EVERY put site, not the first.
        #
        # This used to instrument only the first, with a comment saying one
        # sample was enough. The put appears on two paths -- the partial-tail
        # one at the top of the file and the full-block one further down -- and
        # the arms take the second. So `save_put_count` climbed to 176 and
        # 22 GB landed in the store while the SAVE sample line never once
        # printed, and every log read that day showed a LOOKUP prefix with
        # nothing to compare it against.
        #
        # The whole point of this patch is to put the written key next to the
        # queried key. Instrumenting one arbitrary path of two defeats it
        # silently, which is worse than not having it: the missing line reads
        # like "no saves happened".
        ("save", SAVE_BEFORE, SAVE_AFTER, False),
    ):
        n = src.count(before)
        if n == 0:
            print(
                f"[keysample] FATAL: no {what} anchor. The upstream body "
                f"changed; re-read it before assuming this still applies.",
                file=sys.stderr,
            )
            return 1
        src = src.replace(before, after, 1 if only_first else -1)
        print(f"[keysample]   {what}: instrumented {1 if only_first else n} site(s)")

    path.write_text(src)
    print("[keysample] applied: one sample key logged from each side")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
