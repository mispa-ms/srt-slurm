#!/usr/bin/env bash
# The pdx stack plus the Nsight Systems CLI, in that order.
#
# WHY A COMBINED SCRIPT. `setup_script` takes one entry, and an nsys capture on
# pdx needs both stacks: the profiler (absent from the image -- the worker exits
# 127 without it) and the EFA libfabric + Mooncake wheel that every pdx run
# depends on. Running only the nsys one would strand the store; running only the
# pdx one gives a job that reports success with an empty profiles/ directory.
#
# WHAT THE CAPTURE IS FOR. store.so and engine.so import `cudaMemcpy` and
# `cudaStreamSynchronize` with no `_ptsz` suffix, so the store's device copies
# run on the LEGACY default stream, which implicitly synchronises with every
# blocking stream in the context. The main thread launches CUDA graphs from
# another thread at the same time. The trace should show whether those actually
# serialise, and for how long -- a mechanism question that no amount of A/B at a
# 17% base rate can answer.
#
# The window is placed in WARMUP on purpose. No run has ever stalled there, the
# store's save traffic is heaviest while the cache is cold, and a stall before
# `--duration` expiry would lose the trace.
set -u

echo "=== wei-prebuilt-pdx-nsys: installing Nsight Systems first ==="
bash /configs/patches/install-nsys-cli.sh || {
    echo "wei-prebuilt-pdx-nsys: FATAL: nsys did not install; the worker would" >&2
    echo "  exit 127 at launch. Failing here instead." >&2
    exit 1
}
command -v nsys > /dev/null 2>&1 && nsys --version | head -1

echo "=== wei-prebuilt-pdx-nsys: handing over to the pdx stack ==="
bash /configs/patches/kimi-k3-wei-prebuilt-pdx.sh
