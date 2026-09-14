#!/bin/bash
# The pdx setup, plus: one CUDA event per rank in the store's save fence instead
# of one per step. Same healthy sampling as `-evtctl` and `-evtpool`, so the
# three arms differ only in what they do to the event.
#
# WHY A SEPARATE SCRIPT. The config's `environment:` block is applied AFTER the
# setup script runs, so an env-var gate inside the main script would always read
# unset -- the trap that cost four runs with HF_HOME.
#
# WHAT IS LEFT TO TEST. Everything else on the save path is closed:
#
#   synchronize on a worker thread   `-blockev`, Event(blocking=True)     stalled
#   destroy on a worker thread       `-evtpool`, event ring (431607)      stalled
#   the device copy itself           dsxe 0.3.13.post1+copystream wheel   2/2 stall
#   RDMA read of GPU memory          their tcp arms stall too             not required
#
# The per-step `cudaEventCreateWithFlags` is the one element never removed. This
# arm removes it: one create for the whole run, no destroys.
set -euo pipefail

export PDX_HEALTHY_DUMP_EVERY=30
export PDX_HEALTHY_DUMP_MAX=6

echo "=== evtreuse: one store fence event per rank ==="
python3 /configs/patches/pdx_store_event_reuse.py || {
    echo "wei-prebuilt-pdx: FATAL: the store event-reuse patch did not apply." >&2
    exit 1
}
bash /configs/patches/kimi-k3-wei-prebuilt-pdx.sh
echo "=== evtreuse: done ==="
