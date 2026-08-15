#!/usr/bin/env python3
"""Log the per-group replication factor, put step, and lookup prefix count.

    python3 configs/patches/apply-group-topology-log.py <site-packages>

The save side distributes chunks across ranks:

    put_step      = self.group_put_steps[g_idx]     # = the replication factor
    put_step_rank = (self.tp_rank + g_idx) % put_step
    if block_idx % put_step != put_step_rank: continue

The lookup side, under DCP, requires every rank namespace to hold the key:

    if all(res[pos + j] == 1 for j in range(count)):   # count = 8 here

Those two only agree when put_step is 1. `_spec_tp_replication_factor` returns
1 as soon as `dcp_size > 1`, so on paper it is -- but this is a hybrid model
with several cache groups and the factor is computed per group from its own
spec. A group that reaches the MLA branch instead gets `tp_size`, and then one
rank in eight writes each chunk while the lookup waits for all eight.

Measured: 19,615 individual keys confirmed present by the store across 3.8M
asked, and not one survives the quorum. This prints the two numbers that decide
whether that is the reason, once, at init:

    [mooncake-topology] group=0 spec=MLAAttentionSpec block_size=12288
                        replication_factor=8 put_step=8 lookup_prefixes=8
                        -> one rank in 8 writes each chunk; the lookup needs all 8

put_step > 1 with lookup_prefixes > 1 is the contradiction. put_step == 1
everywhere means the quorum is satisfiable and the shards go missing later,
which is the readback arm's question instead.
"""

import pathlib
import sys

TARGET = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"

BEFORE = """        self.kv_send_thread.start()
"""

AFTER = '''        self.kv_send_thread.start()
        try:
            for _g, _grp in enumerate(self._kv_cache_groups):
                _f = self._group_tp_replication_factors[_g]
                _p = len(self._lookup_key_prefixes[_g])
                logger.info(
                    "[mooncake-topology] group=%d spec=%s block_size=%s "
                    "replication_factor=%d put_step=%d lookup_prefixes=%d -> %s",
                    _g,
                    type(_grp.kv_cache_spec).__name__,
                    getattr(_grp.kv_cache_spec, "block_size", "?"),
                    _f, _f, _p,
                    ("one rank in %d writes each chunk; the lookup needs all %d"
                     % (_f, _p)) if _f > 1 and _p > 1 else "consistent",
                )
        except Exception:
            logger.exception("[mooncake-topology] could not read the group topology")
'''


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[group-topology] FATAL: {path} not found", file=sys.stderr)
        return 1
    src = path.read_text()
    if "mooncake-topology" in src:
        print("[group-topology] already present")
        return 0
    if src.count(BEFORE) != 1:
        print(f"[group-topology] FATAL: {src.count(BEFORE)} anchors, expected 1",
              file=sys.stderr)
        return 1
    path.write_text(src.replace(BEFORE, AFTER, 1))
    print("[group-topology] applied: per-group replication factor and put step logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
