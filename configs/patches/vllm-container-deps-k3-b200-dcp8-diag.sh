#!/usr/bin/env bash
# vllm-container-deps-k3-b200-dcp8.sh plus one read-only delta: say what ranks
# each Mooncake store worker actually holds.
# =============================================================================
# WHY THIS IS NOT A FIX. The lookup side derives its key namespace from a
# relation -- "DCP is a TP subdivision: dcp_rank == tp_rank % dcp_size" -- while
# the store side keys on get_dcp_group().rank_in_group. It is tempting to read
# PR #2618's 1.3% external hit rate (528,384 of 39,923,446, against #2569's 75%
# on the same trace at the same GPU hit rate) as those two disagreeing under
# DP2/EP16, and to widen the lookup to the tp x dcp product.
#
# Our own commit 29961d6bc says otherwise, in as many words: "The refusal was not
# about keys. PoolKey.build_prefix has always carried @dcp{rank}, so each rank's
# strided slice already keys separately. It was about coordinates." Widening the
# lookup would hide the symptom whether or not that reading is right, which is
# the worst outcome available.
#
# So this changes nothing. It logs the actual ranks, the derived one, whether
# they agree, and how many prefixes the lookup built -- once per worker at init.
# One line per rank settles by observation what a hit rate can only suggest.
#
# That same commit ends: "NOT YET MEASURED ... a wrong block size here makes a
# rank load another rank's KV, and synthetic acceptance plus ignore_eos would
# show nothing. This must ship with a GSM8K arm." That arm never ran. It is
# submitted alongside this one.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8.sh

echo "=== k3-b200-dcp8-diag: report the ranks each Mooncake worker actually holds ==="

python3 - <<'PY'
import importlib.util, os, sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
)
src = open(target).read()
if "[dcpdiag]" in src:
    print("[dcpdiag] already applied: " + target)
    sys.exit(0)

anchor = "        self._init_lookup_key_prefixes()"
if src.count(anchor) != 1:
    sys.exit("[dcpdiag] FATAL: anchor found %d times in %s" % (src.count(anchor), target))

addition = anchor + """
        logger.info(
            "[dcpdiag] dp=%s tp=%s/%s dcp=%s/%s pcp=%s/%s pp=%s/%s "
            "derived_dcp=%s agrees=%s lookup_prefixes=%s",
            self.dp_rank, self.tp_rank, self.tp_size,
            self.dcp_rank, self.dcp_size, self.pcp_rank, self.pcp_size,
            self.pp_rank, self.pp_size, self.tp_rank % self.dcp_size,
            self.dcp_rank == self.tp_rank % self.dcp_size,
            sum(len(g) for g in self._lookup_key_prefixes),
        )"""

patched = src.replace(anchor, addition)
compile(patched, target, "exec")
open(target, "w").write(patched)
print("[dcpdiag] applied: " + target)
PY

echo "=== k3-b200-dcp8-diag: done ==="
