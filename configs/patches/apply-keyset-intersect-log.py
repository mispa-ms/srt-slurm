#!/usr/bin/env python3
"""Intersect the keys the store was given with the keys it is asked for.

    python3 configs/patches/apply-keyset-intersect-log.py <site-packages>

Everything else is now ruled out. The instrumented lookup reports

    calls=128 capacity_only=0 no_hashes=0 zero_len=0
    asked=128 raised=0 returned_nonzero=0

so the lookup is called, reaches the store every time, raises nothing, and
comes back empty 128 times out of 128. The key *prefixes* match exactly --
`Kimi-K3@tp_rank:N@pcp0@dcpN@pp_rank:0@group:0` on both sides, all eight ranks
-- so it is not a namespace split either.

What is left is the hash suffix, and the prefix-only sample cannot see it:

    keys[0].rsplit("@", 1)[0]      # drops precisely the part in question

So record what was actually written, and intersect it with what is actually
asked, inside one process:

    [mooncake-keyset] asked=5216 exists=0 ever_saved=0 saved_pool=1112
                      asked_head=<hash> saved_head=<hash>

  ever_saved = 0  -> the two sides derive different hashes from the same
                     tokens. The store is asked for keys nobody wrote, which
                     is what test_mooncake_dcp_keyset.py has claimed at
                     dcp=2/4/8 since the beginning and was dismissed twice.
  ever_saved > 0, exists = 0
                  -> the keys were written and the store denies holding them.
                     A different problem entirely: eviction, tenant, or the
                     `all(res[...])` quorum across rank namespaces.

The saved pool is bounded and per-process; it is a diagnostic, not a cache.
"""

import pathlib
import sys

TARGET = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"

# The deep put -- the one the arms take. Wrapping its arguments one per line is
# what distinguishes it from the partial-tail site.
SAVE_BEFORE = """                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
"""

SAVE_AFTER = '''                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
                _pool = getattr(self, "_ks_saved", None)
                if _pool is None:
                    _pool = self._ks_saved = set()
                if len(_pool) < 400000:
                    _pool.update(keys)
'''

LOOKUP_BEFORE = """            res = self.store.batch_is_exist(candidate_keys)
"""

LOOKUP_AFTER = '''            res = self.store.batch_is_exist(candidate_keys)
            _n = getattr(self, "_ks_probe_n", 0)
            self._ks_probe_n = _n + 1
            if _n % 64 == 0:
                _pool = getattr(self, "_ks_saved", set())
                _ever = sum(1 for k in candidate_keys if k in _pool)
                logger.info(
                    "[mooncake-keyset] asked=%d exists=%d ever_saved=%d "
                    "saved_pool=%d asked_head=%s saved_head=%s",
                    len(candidate_keys),
                    sum(1 for r in res if r == 1),
                    _ever,
                    len(_pool),
                    candidate_keys[0].rsplit("@", 1)[-1] if candidate_keys else "-",
                    next(iter(_pool)).rsplit("@", 1)[-1] if _pool else "-",
                )
'''


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[keyset-intersect] FATAL: {path} not found", file=sys.stderr)
        return 1

    src = path.read_text()
    if "_ks_saved" in src:
        print("[keyset-intersect] already present")
        return 0

    for what, before, after in (("save", SAVE_BEFORE, SAVE_AFTER),
                                ("lookup", LOOKUP_BEFORE, LOOKUP_AFTER)):
        n = src.count(before)
        if n != 1:
            print(f"[keyset-intersect] FATAL: {n} {what} anchors, expected 1. "
                  f"The upstream body changed; re-read it.", file=sys.stderr)
            return 1
        src = src.replace(before, after, 1)

    path.write_text(src)
    print("[keyset-intersect] applied: written keys are intersected with asked keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
