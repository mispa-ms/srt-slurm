#!/bin/bash
# The pdx setup, plus: the store's per-step CUDA event is freed on the main
# thread instead of on the sending thread. Same healthy sampling as `-evtctl`,
# so the two arms differ by exactly one thing.
#
# WHY A SEPARATE SCRIPT. The config's `environment:` block is applied AFTER the
# setup script runs, so an env-var gate inside the main script would always read
# unset -- the same trap that cost four runs with HF_HOME.
#
# THE EVIDENCE THIS TESTS. From the idle rank of run 427249 (GPU 2 at 0%, the
# other seven at 100%), the two threads that are in libcuda:
#
#   KVCacheStoreSendingThread        MainThread
#     pthread_rwlock_wrlock            sched_yield
#     cuEventDestroy_v2                cuGraphLaunch
#     cudaEventDestroy                 cudaGraphLaunch
#     THCPEvent_dealloc                replay (breakable_cudagraph.py:214)
#     run (store/worker.py:614)        run_pw_graph (cudagraph_utils.py:460)
#
# The connector allocates one torch.cuda.Event per step on the main thread
# (worker.py:2159) and hands it to the sending thread, which drops the last
# reference at the top of its loop (worker.py:614, `request_data = None`). So a
# cudaEventDestroy runs off-thread once per saved step, concurrent with the main
# thread's graph launches -- and it exists only when there is something to save,
# which is the necessary condition every A/B has found.
#
# THE PATCH moves the free, not the semantics: the main thread keeps each event
# in a bounded ring, so eviction and the destroy happen there.
#
# BASELINE. This exact configuration, without the patch, stalled in 6 of 6 runs.
set -euo pipefail

export PDX_HEALTHY_DUMP_EVERY=30
export PDX_HEALTHY_DUMP_MAX=6
export PDX_STORE_EVENT_POOL=1
export PDX_STORE_EVENT_RING=256

echo "=== evtpool: store event ring ${PDX_STORE_EVENT_RING}," \
     "healthy sampling every ${PDX_HEALTHY_DUMP_EVERY} ticks ==="
bash /configs/patches/kimi-k3-wei-prebuilt-pdx.sh
echo "=== evtpool: done ==="
