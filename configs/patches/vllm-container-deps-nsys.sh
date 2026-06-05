#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same as vllm-container-deps.sh, plus Nsight Systems CLI for profiling-enabled
# vLLM recipes. The base vllm/vllm-openai images don't ship nsys, so worker
# launches that wrap with `nsys profile ...` exit 127.

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# Nsight Systems. The vllm/vllm-openai image already has the NVIDIA cuda repo
# registered, so a direct `apt install` works. Installing cuda-keyring on top
# triggers an apt Signed-By conflict.
#
# `nsight-systems-cli` is pinned to 2024.2.3 on ubuntu2404/sbsa, which lacks
# `--gpu-metrics-devices`. The full `nsight-systems-2025.6.3` package carries
# the version we need (confirmed via the repo Packages index for
# ubuntu2404/sbsa).
if ! command -v nsys >/dev/null 2>&1; then
    apt-get -y update
    apt-get install -y --no-install-recommends nsight-systems-2025.6.3
fi
nsys --version

# ----------------------------------------------------------------------------
# Patch vLLM's gpu_worker.execute_dummy_batch to tick the WorkerProfiler step
# counter on 0-token iters, restoring symmetry with execute_model.
#
# Why this is needed:
#   `wrapper.step()` advances `_active_iteration_count` and triggers
#   cudaProfilerStart after `delay_iterations`. It is only called from
#   `gpu_worker.annotate_profile`, which wraps `model_runner.execute_model`.
#   On 0-token iters in multiproc DP mode, `DPEngineCoreProc.run_busy_loop`
#   dispatches `execute_dummy_batch` instead, and that path calls
#   `_dummy_run` directly without `annotate_profile`. 0-token iters
#   therefore skip the step tick.
#
# Consequence: ranks with more tokens during warmup tick the counter
# more often than 0-token ranks → `delay_iters=500` is reached at
# different wall-clock times across ranks → cudaProfilerStart fires at
# different wall-clock times across DP ranks → DeepGEMM mega_moe
# nvlink_barrier counter drifts post-profile → post-stop cascade
# (EngineDeadError on a decode rank seconds after cudaProfilerStop).
#
# The fix: tick wrapper.step() inside execute_dummy_batch as well, so
# every iter (active or dummy) advances the counter exactly once.
# wrapper.step() is a pure CPU op (no NCCL, no CUDA collective).
#
# Idempotent via string presence check.
python3 - <<'PYPATCH_DUMMY_TICK'
import pathlib, sys

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
        # counter stuck. wrapper.step() itself is a pure CPU op (no NCCL /
        # CUDA collective).
        if self.profiler is not None:
            self.profiler.step()
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)"""

if new_block in content:
    print(f"[vllm-dummy-batch-tick-patch] Already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    print("[vllm-dummy-batch-tick-patch] Expected block not found - vLLM source drift?")
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
#     1. annotate_profile is called → would tick wrapper.step() (this
#        gate blocks it when tokens == 0)
#     2. execute_model hits an early return, runs kv_connector_no_forward
#     3. EngineCore.step() returns model_executed=False
#     4. DPEngineCoreProc.run_busy_loop sees not-executed → calls
#        execute_dummy_batch → PYPATCH_DUMMY_TICK ticks wrapper.step()
#   Without this gate, wrapper.step() would tick TWICE on KV-only iters,
#   reintroducing per-rank asymmetry (token-poor ranks have more KV-only
#   iters).
#
# With BOTH PYPATCH_DUMMY_TICK and this gate, every iter type ticks
# exactly once across all DP ranks:
#   - Active token iter (tokens > 0): annotate_profile ticks (gate passes)
#   - 0-token + KV iter:              dummy ticks (annotate_profile gated)
#   - Pure-dummy iter:                dummy ticks
#
# Idempotent.
python3 - <<'PYPATCH_ANNOTATE_GATE'
import pathlib, sys

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
        # tick on tokens > 0. Pairs with PYPATCH_DUMMY_TICK so that every
        # iter (active / KV-only / pure-dummy) ticks exactly once across DP
        # ranks. Without this gate, KV-only iters would tick twice — once
        # here, once via execute_dummy_batch fired by run_busy_loop because
        # model_executed=False — reintroducing per-rank counter asymmetry.
        if scheduler_output.total_num_scheduled_tokens > 0:
            self.profiler.step()"""

if new_block in content:
    print(f"[vllm-annotate-profile-gate-patch] Already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    print("[vllm-annotate-profile-gate-patch] Expected block not found - vLLM source drift?")
    print("[vllm-annotate-profile-gate-patch] File:", target)
    sys.exit(0)

content = content.replace(old_block, new_block)
target.write_text(content)
print(f"[vllm-annotate-profile-gate-patch] Patched (gated wrapper.step() on tokens > 0) in {target}")
PYPATCH_ANNOTATE_GATE
