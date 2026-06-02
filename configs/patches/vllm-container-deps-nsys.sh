#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same as vllm-container-deps.sh, plus Nsight Systems CLI for profiling-enabled
# vLLM recipes (e.g. disagg-1p1d-dep4-dep16-decode-only-c4096-nsys.yml).
# The base vllm/vllm-openai images don't ship nsys, so worker launches that
# wrap with `nsys profile ...` exit 127.

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# Nsight Systems. The vllm/vllm-openai image already has the NVIDIA cuda
# repo registered, so a direct `apt install` works. (Installing cuda-keyring
# on top triggers an apt Signed-By conflict — seen on #53132895, #53135432.)
#
# The `nsight-systems-cli` package is pinned to 2024.2.3 on ubuntu2404/sbsa,
# which lacks `--gpu-metrics-devices` entirely (rejected as unrecognised on
# #53136239). Switch to `nsight-systems-2025.6.3` (the full nsight-systems
# package family carries newer versions — confirmed via the repo Packages
# index for ubuntu2404/sbsa).
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    apt-get install -y --no-install-recommends nsight-systems-2025.6.3
fi
nsys --version

# Patch vLLM's CudaProfilerWrapper to (a) call torch.cuda.synchronize() and
# (b) issue torch.distributed.barrier() before cudaProfilerStart/Stop.
#
# Why sync: without it, the stop path fires cudaProfilerStop while kernels
# are still in flight inside the FULL_DECODE_ONLY cudagraph vLLM replays
# every decode step. CUPTI's stop interaction with an in-flight graph
# leaves the CUDA context in a broken state — a few seconds later the
# next sample_tokens.async_copy_ready_event.synchronize() returns
# cudaErrorLaunchFailure on the profiled rank and decode cascades.
#
# Why barrier: SGLang's profile_utils.py wraps every cudaProfilerStart/Stop
# in torch.distributed.barrier(cpu_group) so all DP ranks transition CUPTI
# state atomically. vLLM has no equivalent. Per-rank _start/_stop fires at
# slightly different wall-clock times — worker busy-loop dequeue from
# rpc_broadcast_mq has ~10s-of-ms jitter, and in iteration-driven mode the
# per-rank step counter race adds seconds (53s observed on #53246692).
# The asymmetric profile-active window during NCCL all-reduce in
# coordinate_batch_across_dp produces cross-rank CUPTI state divergence
# that surfaces minutes later as a launch failure.
#
# A previous attempt (srt-slurm 65b85db) added this exact barrier and was
# reverted on 92ac1b3 because iteration-driven mode (max_iterations=50)
# made ranks reach _stop() at different step counts — one rank entered
# _stop() and waited at the barrier while another rank never reached the
# threshold, full-group deadlock on #53249727. RPC-driven mode
# (PROFILE_RPC_DURATION_SEC env in bench.sh + profiler-config without
# max_iterations) bypasses the per-rank step counter entirely: the
# /engine/start_profile RPC arrives at all ranks within ms via
# rpc_broadcast_mq broadcast, _start fires at the busy-loop boundary, and
# all ranks reach the barrier within ms. Barrier converges; no deadlock.
#
# Apply via Python AST-safe rewrite — the wrapper.py override block has a
# stable shape since vLLM 0.21.0.
python3 - <<'PYPATCH'
import pathlib, sys
candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/profiler/wrapper.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/profiler/wrapper.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[vllm-cuda-profiler-sync-patch] wrapper.py not found, skipping")
    sys.exit(0)

content = target.read_text()
old_start = """    @override
    def _start(self) -> None:
        self._cuda_profiler.start()"""
new_start = """    @override
    def _start(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._cuda_profiler.start()"""
old_stop = """    @override
    def _stop(self) -> None:
        self._cuda_profiler.stop()"""
new_stop = """    @override
    def _stop(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._cuda_profiler.stop()"""

if "torch.distributed.barrier()" in content:
    print("[vllm-cuda-profiler-sync-patch] Already patched (sync + barrier), skipping.")
    sys.exit(0)
if old_start not in content or old_stop not in content:
    # Maybe already patched with sync-only (prior srt-slurm 92ac1b3 baseline).
    sync_only_start = """    @override
    def _start(self) -> None:
        import torch
        torch.cuda.synchronize()
        self._cuda_profiler.start()"""
    sync_only_stop = """    @override
    def _stop(self) -> None:
        import torch
        torch.cuda.synchronize()
        self._cuda_profiler.stop()"""
    if sync_only_start in content and sync_only_stop in content:
        content = content.replace(sync_only_start, new_start).replace(sync_only_stop, new_stop)
        target.write_text(content)
        print("[vllm-cuda-profiler-sync-patch] Upgraded sync-only -> sync + barrier in", target)
        sys.exit(0)
    print("[vllm-cuda-profiler-sync-patch] Expected blocks not found — vLLM source drift?")
    print("[vllm-cuda-profiler-sync-patch] File:", target)
    sys.exit(0)

content = content.replace(old_start, new_start).replace(old_stop, new_stop)
target.write_text(content)
print("[vllm-cuda-profiler-sync-patch] Patched CudaProfilerWrapper._start/_stop (sync + barrier) in", target)
PYPATCH
