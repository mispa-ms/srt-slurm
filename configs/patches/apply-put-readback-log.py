#!/usr/bin/env python3
"""Ask the store for the keys it just accepted.

    python3 configs/patches/apply-put-readback-log.py <site-packages>

The save path and the lookup path call the same primitive:

    exists = self.store.batch_is_exist(keys)     # save side, before writing
    res    = self.store.batch_is_exist(candidate_keys)   # lookup side

The lookup side answers "not present" for every key -- 112,064 of 112,064 in a
single call -- while the workers write 22 GB with save_put_failed_keys at 0 and
no exception anywhere. Prefixes match on both sides across all eight DCP ranks,
the save's key index falls inside the range the lookup enumerates, and the
lookup reaches the store every time.

Everything in that list is about *which* keys. This tests something else: put a
batch, then immediately ask for exactly the keys just put, in the same process,
on the same store handle, with the same strings.

    [mooncake-readback] put=8 put_ok=8 readback_present=0/8 key=<whole key>

  readback_present = 0  -> the store accepts a write and denies holding it one
                           call later. Key derivation, alignment and namespaces
                           are all beside the point; the fault is the store
                           handle or its configuration -- tenant, segment
                           registration, or the master this client talks to.
  readback_present = 8  -> writes are visible, so the lookup is asking for
                           different strings after all, and the whole-key
                           comparison is where to look.

One batch is enough, so it fires once per process.
"""

import pathlib
import sys

TARGET = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"

# The deep put -- the one the arms take. Its arguments are wrapped one per line,
# which is what tells it apart from the partial-tail site.
BEFORE = """                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
"""

AFTER = '''                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
                if keys and not getattr(self, "_rb_done", False):
                    self._rb_done = True
                    try:
                        _ok = sum(1 for r in res if r == 0) if res is not None else -1
                        _back = self.store.batch_is_exist(list(keys))
                        logger.info(
                            "[mooncake-readback] put=%d put_ok=%s "
                            "readback_present=%d/%d key=%s",
                            len(keys),
                            _ok,
                            sum(1 for r in _back if r == 1),
                            len(_back),
                            keys[0],
                        )
                    except Exception:
                        logger.exception("[mooncake-readback] readback raised")
'''


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[put-readback] FATAL: {path} not found", file=sys.stderr)
        return 1

    src = path.read_text()
    if "_rb_done" in src:
        print("[put-readback] already present")
        return 0
    if src.count(BEFORE) != 1:
        print(f"[put-readback] FATAL: {src.count(BEFORE)} anchors, expected 1",
              file=sys.stderr)
        return 1

    path.write_text(src.replace(BEFORE, AFTER, 1))
    print("[put-readback] applied: keys are re-queried immediately after the put")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
