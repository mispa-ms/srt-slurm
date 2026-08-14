#!/usr/bin/env python3
"""Align the mooncake key lookup to what the store can answer.

Not a .patch file, deliberately. Two attempts at one failed for reasons that
have nothing to do with the change: the first was regenerated against
repo/vllm, whose ``attention_groups`` unpacking has three elements where the
patched nightly has four, and the second was hand-written with a bare ``@@``
header that GNU patch rejects outright. Both cost a full submit to discover. A
string replacement with an assertion cannot drift with context or line numbers,
and says exactly what it did.

The change: ``align_lookup_length`` decides which keys are asked for --
worker.py fixes ``token_len`` from it and then builds the candidate key list.
The store writes one key per ``lcm_block_size`` chunk, so those are the only
lengths it can serve. Aligning to ``hash_block_size`` instead lets a hit land
between two written keys, and the load then asks for a key nobody wrote: the
request succeeds, finds nothing, and reports no error. Under DCP8 a block is
12288 tokens against a 128-token hash block.

The sibling change, to the alignment that decides how much of a hit is
*reported*, is in vllm-mooncake-external-hit-block-aligned.patch. Fixing either
alone leaves the other half of the pair.
"""

import pathlib
import sys

TARGET = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py"

# coordinator.py has no logger. worker.py defines one; this module never needed
# it until the line below was added, and `logger.info` there raises NameError --
# inside the connector's process_request thread, so the thread dies, the lookup
# never returns, and the client sits at "51 in flight, 0 returned, 0 errors"
# until the idle-GPU reaper claims the job. Four ladders were lost to that and
# read as a reaper problem. Import it explicitly rather than assume.
LOGGER_IMPORT = "from vllm.logger import init_logger\n\nlogger = init_logger(__name__)\n"

BEFORE = """    def align_lookup_length(self, length: int) -> int:
        alignment = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        return length // alignment * alignment
"""

AFTER = '''    def align_lookup_length(self, length: int) -> int:
        # This alignment decides which keys are asked for: worker.py fixes
        # token_len from it and builds the candidate key list. The one in
        # find_longest_cache_hit only decides how much of a hit is reported.
        alignment = self.lcm_block_size
        aligned = length // alignment * alignment
        # Once per process: does the divergence this assumes actually exist on
        # this configuration? Under DCP8 a block is 12288 tokens against a
        # 128-token hash block, and a hit landing between two written keys is
        # what makes the offload tier write gigabytes and read back 0%.
        if not getattr(self, "_logged_keyset_alignment", False):
            self._logged_keyset_alignment = True
            logger.info(
                "[mooncake-keyset] lookup aligns to %d; hash_block_size=%d, "
                "lcm_block_size=%d, partial_hash_hits=%s; upstream would use "
                "%d, and the store answers only multiples of %d",
                alignment,
                self.hash_block_size,
                self.lcm_block_size,
                self.enable_partial_hash_hits,
                self.hash_block_size
                if self.enable_partial_hash_hits
                else self.lcm_block_size,
                self.lcm_block_size,
            )
        return aligned
'''


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[lookup-align] FATAL: {path} not found", file=sys.stderr)
        return 1

    src = path.read_text()
    if "_logged_keyset_alignment" in src:
        print("[lookup-align] already present")
        return 0

    # The logger first, and verified: without it the added log line raises
    # NameError in a worker thread and the stall is silent.
    if "init_logger" not in src:
        marker = "\n\nclass MooncakeStoreCoordinator"
        if marker not in src:
            print("[lookup-align] FATAL: cannot place the logger import",
                  file=sys.stderr)
            return 1
        src = src.replace(marker, "\n\n" + LOGGER_IMPORT + marker, 1)
    if src.count(BEFORE) != 1:
        print(
            f"[lookup-align] FATAL: align_lookup_length does not match the "
            f"expected form ({src.count(BEFORE)} matches). The upstream body "
            f"changed; re-read it before assuming this change still applies.",
            file=sys.stderr,
        )
        return 1

    path.write_text(src.replace(BEFORE, AFTER))
    print("[lookup-align] applied: lookup aligns to lcm_block_size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
