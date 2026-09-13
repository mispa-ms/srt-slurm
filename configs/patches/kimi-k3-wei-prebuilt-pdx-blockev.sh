#!/bin/bash
# The pdx setup, plus one line of vLLM: make the Mooncake save event a sleeping
# event instead of a spinning one.
#
# WHY A SEPARATE SCRIPT. The config's `environment:` block is applied AFTER the
# setup script runs, so an env-var gate inside the main script would always read
# unset. `setup_script:` is a config field, so pointing a variant at this file is
# the only clean way to A/B a source patch.
#
# THE PATCH. vLLM's own async-output path documents the hazard and avoids it:
#
#   vllm/v1/worker/gpu/async_utils.py:33
#     # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
#     self.copy_event = torch.cuda.Event(blocking=True)
#
# The Mooncake store's save path does the opposite -- a plain torch.cuda.Event,
# so cudaEventBlockingSync is unset and cudaEventSynchronize spins -- and a
# background thread then waits on it for the whole forward, every step, on every
# rank with something to save (store/worker.py:641 and :859).
#
# Candidate mechanism for the stall that has cost us seven runs and b300-dsxe
# twenty-one: driver-lock starvation. One rank's main thread cannot push its
# launches through while a sibling thread spins on the lock, so it never joins
# the collective; the other seven sit in it at 100% waiting for a rank at 0%.
# Fits the save-only trigger, the fabric independence, the absence of Xid, the
# moving victim, and breakable-off shifting the odds without fixing anything.
set -euo pipefail

bash /configs/patches/kimi-k3-wei-prebuilt-pdx.sh

echo "=== blockev: making the Mooncake save event a sleeping event ==="
python3 - <<'PYBLOCKEV'
import pathlib, sys

p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/"
    "kv_connector/v1/mooncake/store/worker.py"
)
if not p.exists():
    sys.exit(f"blockev: FATAL: {p} does not exist; the store connector moved.")

t = p.read_text()
old = "current_event = torch.cuda.Event()"
new = "current_event = torch.cuda.Event(blocking=True)"

if new in t:
    print("    already patched")
    sys.exit(0)

n = t.count(old)
# Fail loudly. A silent no-op here would produce a run that looks like a clean
# A/B and is in fact the control.
if n != 1:
    sys.exit(f"blockev: FATAL: expected exactly one {old!r}, found {n}. "
             "Upstream changed; re-read store/worker.py before trusting this arm.")

p.write_text(t.replace(old, new, 1))
print(f"    patched {p}")

check = p.read_text()
assert new in check and check.count(old) == 0, "blockev: verification failed"
print("    verified: torch.cuda.Event(blocking=True)")
PYBLOCKEV

echo "=== blockev: done ==="
