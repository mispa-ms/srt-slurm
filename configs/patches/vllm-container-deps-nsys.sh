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
#
# WRAPPER_VARIANT env var selects the wrapper variant to install:
#   default     : sync + dist.barrier(device_ids=[current_device])     [v17 baseline]
#   dp_barrier  : sync + barrier on vllm DP group (lazy import)         [v19 experiment]
#   double_sync : sync + barrier(device_ids) + sync again before        [v20 experiment]
#                 cudaProfilerStart (drain barrier's NCCL kernel)
#
# v17 evidence (#53454633): default variant cleared the barrier
# itself (no "Guessing device ID" warning), but a SUBSEQUENT
# coordinate_batch_across_dp ALLREDUCE on the DP group (PG 6,
# NumelIn=16 matches dp_utils.py:47 torch.zeros(4, dp_size)) hung
# for 600s and tripped the NCCL watchdog. Our wrapper barrier is on
# default_pg (PG 0) — a different NCCL communicator than the DP
# group that hangs. dp_barrier and double_sync are two competing
# hypotheses for why PG 6 hangs after our default-pg barrier:
#   - dp_barrier  : the right group to sync is DP group, not default
#   - double_sync : barrier's own NCCL kernel is still in flight when
#                   cudaProfilerStart fires, CUPTI's hooks land on a
#                   busy stream and corrupt the DP group's communicator
WRAPPER_VARIANT="${WRAPPER_VARIANT:-default}"
echo "[vllm-cuda-profiler-sync-patch] WRAPPER_VARIANT=${WRAPPER_VARIANT}"

WRAPPER_VARIANT="${WRAPPER_VARIANT}" python3 - <<'PYPATCH'
import os, pathlib, sys

variant = os.environ.get("WRAPPER_VARIANT", "default")

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/profiler/wrapper.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/profiler/wrapper.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[vllm-cuda-profiler-sync-patch] wrapper.py not found, skipping")
    sys.exit(0)

content = target.read_text()

# Pristine source blocks (what vLLM v0.21.0 ships).
old_start = """    @override
    def _start(self) -> None:
        self._cuda_profiler.start()"""
old_stop = """    @override
    def _stop(self) -> None:
        self._cuda_profiler.stop()"""

# Variant: default — sync + barrier(device_ids).
default_start = """    @override
    def _start(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        self._cuda_profiler.start()"""
default_stop = """    @override
    def _stop(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        self._cuda_profiler.stop()"""

# Variant: dp_barrier — barrier on vLLM DP group, fall back to default.
dp_barrier_start = """    @override
    def _start(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from vllm.distributed.parallel_state import get_dp_group
                _dp_grp = get_dp_group().device_group
            except Exception:
                _dp_grp = None
            if _dp_grp is not None:
                torch.distributed.barrier(group=_dp_grp, device_ids=[torch.cuda.current_device()])
            else:
                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        self._cuda_profiler.start()"""
dp_barrier_stop = """    @override
    def _stop(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from vllm.distributed.parallel_state import get_dp_group
                _dp_grp = get_dp_group().device_group
            except Exception:
                _dp_grp = None
            if _dp_grp is not None:
                torch.distributed.barrier(group=_dp_grp, device_ids=[torch.cuda.current_device()])
            else:
                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        self._cuda_profiler.stop()"""

# Variant: double_sync — default + extra sync between barrier and
# cudaProfilerStart so the barrier's NCCL kernel finishes before
# CUPTI activation lands.
double_sync_start = """    @override
    def _start(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.synchronize()
        self._cuda_profiler.start()"""
double_sync_stop = """    @override
    def _stop(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.synchronize()
        self._cuda_profiler.stop()"""

if variant == "dp_barrier":
    new_start, new_stop = dp_barrier_start, dp_barrier_stop
    marker = "from vllm.distributed.parallel_state import get_dp_group"
    label = "sync + DP-group barrier"
elif variant == "double_sync":
    new_start, new_stop = double_sync_start, double_sync_stop
    # Distinct marker: TWO torch.cuda.synchronize() calls in _start.
    marker = None  # detected via prior variants below
    label = "sync + barrier + sync"
else:
    new_start, new_stop = default_start, default_stop
    marker = "torch.distributed.barrier(device_ids=["
    label = "sync + barrier(device_ids)"

# Idempotency: skip if exact target is already in file.
if new_start in content and new_stop in content:
    print(f"[vllm-cuda-profiler-sync-patch] Already patched ({label}), skipping.")
    sys.exit(0)

# Known prior forms: pristine, sync-only, device-less barrier, default
# (sync + device_ids barrier), dp_barrier, double_sync — try each in
# turn; whichever matches both blocks gets replaced with the chosen
# variant.
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
device_less_start = """    @override
    def _start(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._cuda_profiler.start()"""
device_less_stop = """    @override
    def _stop(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._cuda_profiler.stop()"""

for from_start, from_stop, from_label in (
    (old_start, old_stop, "pristine"),
    (sync_only_start, sync_only_stop, "sync-only"),
    (device_less_start, device_less_stop, "device-less barrier"),
    (default_start, default_stop, "default (sync + device_ids barrier)"),
    (dp_barrier_start, dp_barrier_stop, "dp_barrier"),
    (double_sync_start, double_sync_stop, "double_sync"),
):
    if from_start in content and from_stop in content:
        content = content.replace(from_start, new_start).replace(from_stop, new_stop)
        target.write_text(content)
        print(f"[vllm-cuda-profiler-sync-patch] Patched ({from_label} -> {label}) in {target}")
        sys.exit(0)

print("[vllm-cuda-profiler-sync-patch] Expected blocks not found — vLLM source drift?")
print("[vllm-cuda-profiler-sync-patch] File:", target)
sys.exit(0)
PYPATCH
