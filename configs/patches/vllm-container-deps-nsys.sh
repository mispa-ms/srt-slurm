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

# Variant: cpu_barrier — mirror SGLang's exact pattern. Use the DP group's
# CPU sub-communicator (gloo backend) so the barrier does NOT enqueue a
# NCCL kernel on the same CUDA stream as the in-flight cudagraph replays.
# Prior variants (default, dp_barrier, double_sync) all used NCCL barriers
# on the device, which compete with the running cudagraph for stream
# ordering — that's the underlying cause of the sample_tokens RPC timeout
# we keep seeing. The CPU barrier is exactly what SGLang's profile_utils.py
# does on `dist.barrier(cpu_group)` (line 313, 385 in sglang/srt/utils/
# profile_utils.py).
cpu_barrier_start = """    @override
    def _start(self) -> None:
        import torch
        # Drain in-flight CUDA work on this rank before snapping the profiler.
        torch.cuda.synchronize()
        # Cross-rank synchronize on the DP group's gloo (CPU) communicator
        # so cudaProfilerStart fires simultaneously across all DP ranks
        # without queueing any CUDA-side collective.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from vllm.distributed.parallel_state import get_dp_group
                _cpu_grp = get_dp_group().cpu_group
            except Exception:
                _cpu_grp = None
            torch.distributed.barrier(group=_cpu_grp)
        self._cuda_profiler.start()"""
cpu_barrier_stop = """    @override
    def _stop(self) -> None:
        import torch
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from vllm.distributed.parallel_state import get_dp_group
                _cpu_grp = get_dp_group().cpu_group
            except Exception:
                _cpu_grp = None
            torch.distributed.barrier(group=_cpu_grp)
        self._cuda_profiler.stop()"""

# Variant: rank0_only — match SGLang's actual pattern (profiler_manager.py:224).
# SGLang gates cudaProfilerStart on `gpu_id == base_gpu_id`, i.e. only the
# one process per node whose first visible GPU is rank 0 calls into CUPTI.
# Other ranks set a flag but never touch cudaProfilerStart.
# Why this works where every prior variant failed:
#   * No barrier => no cross-rank deadlock against DP allreduce
#   * Single caller per node => per-rank step counter drift is irrelevant;
#     cross-rank races on CUPTI state can't happen because peer ranks
#     never enter the start path
#   * Trace covers 1 GPU per node, which is exactly what SGLang produces
#     and is sufficient for DLB lane_0 analysis.
rank0_only_start = """    @override
    def _start(self) -> None:
        import os
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        self._cuda_profiler.start()"""
rank0_only_stop = """    @override
    def _stop(self) -> None:
        import os
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
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
elif variant == "cpu_barrier":
    new_start, new_stop = cpu_barrier_start, cpu_barrier_stop
    marker = "get_dp_group().cpu_group"
    label = "sync + DP CPU-group (gloo) barrier — SGLang pattern"
elif variant == "rank0_only":
    new_start, new_stop = rank0_only_start, rank0_only_stop
    marker = 'os.environ.get("LOCAL_RANK", "0")'
    label = "rank0-only cudaProfilerStart (SGLang gpu_id==base_gpu_id pattern)"
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
    (cpu_barrier_start, cpu_barrier_stop, "cpu_barrier"),
    (rank0_only_start, rank0_only_stop, "rank0_only"),
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

# ----------------------------------------------------------------------------
# Patch vLLM's gpu_worker.execute_dummy_batch to tick the WorkerProfiler
# step counter on 0-token iters, restoring symmetry with execute_model.
#
# Why this is needed:
#   wrapper.step() (which advances `_active_iteration_count` and triggers
#   cudaProfilerStart after `delay_iterations`) is only called from
#   `gpu_worker.annotate_profile`, which is wrapped around
#   `model_runner.execute_model` (gpu_worker.py:840). On 0-token iters
#   in multiproc DP mode, `DPEngineCoreProc.run_busy_loop` (core.py:1858)
#   dispatches `execute_dummy_batch` RPC instead of `execute_model` — and
#   `execute_dummy_batch` calls `_dummy_run` directly without
#   `annotate_profile`. So 0-token iters skip the wrapper.step() tick.
#
# Consequence: ranks with more tokens during warmup tick the counter
# more often than 0-token ranks → `delay_iters=500` is reached at
# different wall-clock times across ranks (36 s spread observed on
# v25 #53594803) → cudaProfilerStart fires at different wall-clock
# times across DP ranks → DeepGEMM mega_moe nvlink_barrier counter
# drifts post-profile → cascade.
#
# The fix: tick wrapper.step() inside execute_dummy_batch as well so
# every iter (token or dummy) advances the counter. wrapper.step() is
# a pure CPU op (no NCCL, no CUDA collective) — just increments
# `_active_iteration_count` and conditionally calls _call_start when
# the threshold is reached. So it adds no extra collective.
#
# This is the symmetric counterpart of the existing
# `annotate_profile.profiler.step()` call at gpu_worker.py:755 and
# does NOT need any patch to execute_model itself (v27-dpdummy's
# mistake — that introduced an EXTRA `_dummy_run` call which
# duplicated the coordinate_batch_across_dp allreduce and caused
# SeqNum=35 mismatch on #53615976).
#
# Always applied (defensive). Idempotent via string presence check.
python3 - <<'PYPATCH_DUMMY_TICK'
import os, pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_worker.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[vllm-dummy-batch-tick-patch] gpu_worker.py not found, skipping")
    sys.exit(0)

content = target.read_text()

old_block = """    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)"""

new_block = """    def execute_dummy_batch(self) -> None:
        # [PATCHED by vllm-container-deps-nsys.sh] Tick WorkerProfiler step
        # counter on 0-token iters so `delay_iterations`-based profile trigger
        # fires synchronously across DP ranks. Without this, wrapper.step() is
        # only ticked via annotate_profile (which wraps execute_model), and
        # 0-token iters reach execute_dummy_batch instead — leaving the
        # counter stuck. Token-distribution asymmetry across DP ranks then
        # produces per-rank counter drift → cudaProfilerStart staggers →
        # DeepGEMM mega_moe barrier counter mismatch post-profile → cascade.
        # wrapper.step() itself is a pure CPU op (no NCCL / CUDA collective).
        if self.profiler is not None:
            self.profiler.step()
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)"""

if new_block in content:
    print(f"[vllm-dummy-batch-tick-patch] Already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    print("[vllm-dummy-batch-tick-patch] Expected block not found — vLLM source drift?")
    print("[vllm-dummy-batch-tick-patch] File:", target)
    sys.exit(0)

content = content.replace(old_block, new_block)
target.write_text(content)
print(f"[vllm-dummy-batch-tick-patch] Patched (added wrapper.step() to execute_dummy_batch) in {target}")
PYPATCH_DUMMY_TICK

# ----------------------------------------------------------------------------
# Patch vLLM's gpu_worker.annotate_profile to gate the wrapper.step() tick
# on `total_num_scheduled_tokens > 0`, completing the symmetric counter
# update across all DP iter types.
#
# Why this is needed (edge case discovered after PYPATCH_DUMMY_TICK):
#   execute_model is called by EngineCore.step() whenever
#   `scheduler.has_requests()` is True — including iters where
#   `total_num_scheduled_tokens == 0` but KV-connector work is pending
#   (NIXL KV-transfer iter in disagg decode). In those iters:
#     1. annotate_profile is called → ticks wrapper.step() (this patch removes
#        the tick by gating on tokens > 0)
#     2. execute_model hits early return at gpu_model_runner.py:4007,
#        runs kv_connector_no_forward, returns
#     3. EngineCore.step() returns model_executed=False
#     4. DPEngineCoreProc.run_busy_loop sees not-executed → calls
#        execute_dummy_batch → PYPATCH_DUMMY_TICK ticks wrapper.step()
#   Without this gate: wrapper.step() would tick TWICE on KV-only iters,
#   reintroducing per-rank asymmetry (token-poor ranks have more KV-only
#   iters and thus tick more).
#
# With BOTH PYPATCH_DUMMY_TICK and this gate:
#   - Active token iter (tokens > 0): annotate_profile ticks (gate passes),
#     dummy not called → 1 tick
#   - 0-token + KV iter: annotate_profile skips tick (gated), dummy ticks
#     → 1 tick
#   - Pure-dummy iter (no requests at all): step() early returns without
#     calling execute_model, dummy ticks → 1 tick
#   Every iter type has exactly 1 wrapper.step() tick across all DP ranks.
#
# Always applied. Idempotent.
python3 - <<'PYPATCH_ANNOTATE_GATE'
import os, pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_worker.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[vllm-annotate-profile-gate-patch] gpu_worker.py not found, skipping")
    sys.exit(0)

content = target.read_text()

old_block = """    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:
            return nullcontext()

        self.profiler.step()"""

new_block = """    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:
            return nullcontext()

        # [PATCHED by vllm-container-deps-nsys.sh] Gate the wrapper.step()
        # tick on tokens > 0. This pairs with PYPATCH_DUMMY_TICK
        # (execute_dummy_batch always ticks). Without this gate, KV-only
        # iters (tokens=0 but kv_connector work pending) tick twice —
        # once here, once via execute_dummy_batch fired from
        # DPEngineCoreProc.run_busy_loop because model_executed=False —
        # reintroducing per-rank counter asymmetry.
        if scheduler_output.total_num_scheduled_tokens > 0:
            self.profiler.step()"""

if new_block in content:
    print(f"[vllm-annotate-profile-gate-patch] Already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    print("[vllm-annotate-profile-gate-patch] Expected block not found — vLLM source drift?")
    print("[vllm-annotate-profile-gate-patch] File:", target)
    sys.exit(0)

content = content.replace(old_block, new_block)
target.write_text(content)
print(f"[vllm-annotate-profile-gate-patch] Patched (gated wrapper.step() on tokens > 0) in {target}")
PYPATCH_ANNOTATE_GATE
