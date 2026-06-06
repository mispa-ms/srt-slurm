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
    # Fail hard rather than silently no-op. A silent skip on source drift
    # would let the asymmetric-tick cascade quietly re-emerge on a newer
    # vLLM image — the bench would still run, profile would still fire,
    # and the post-stop EngineDeadError would just come back. Failing the
    # container init forces a "update this patch for the new vLLM shape"
    # signal at the right time.
    print("[vllm-dummy-batch-tick-patch] ERROR: expected block not found in", target)
    print("[vllm-dummy-batch-tick-patch] vLLM source has drifted past v0.21.0 — update PYPATCH_DUMMY_TICK")
    print("[vllm-dummy-batch-tick-patch] in configs/patches/vllm-container-deps-nsys.sh to match the new shape.")
    sys.exit(1)

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
    # Hard fail on source drift — see the matching block in PYPATCH_DUMMY_TICK.
    print("[vllm-annotate-profile-gate-patch] ERROR: expected block not found in", target)
    print("[vllm-annotate-profile-gate-patch] vLLM source has drifted past v0.21.0 — update PYPATCH_ANNOTATE_GATE")
    print("[vllm-annotate-profile-gate-patch] in configs/patches/vllm-container-deps-nsys.sh to match the new shape.")
    sys.exit(1)

content = content.replace(old_block, new_block)
target.write_text(content)
print(f"[vllm-annotate-profile-gate-patch] Patched (gated wrapper.step() on tokens > 0) in {target}")
PYPATCH_ANNOTATE_GATE

# ----------------------------------------------------------------------------
# Patch vLLM's DPEngineCoreProc.run_busy_loop to emit an `Iteration(N)` log
# line for pure-dummy iters too, unifying the engine-side iteration counter
# with the worker-side wrapper.step() counter.
#
# Why this is needed:
#   With PYPATCH_DUMMY_TICK and PYPATCH_ANNOTATE_GATE applied, the
#   worker-side wrapper.step() counter ticks exactly once per iter for ALL
#   iter types (active, KV-only, pure-dummy). But the engine-side
#   `Iteration(N)` log only emits when `EngineCore.step()` runs, which
#   early-returns when `scheduler.has_requests() == False` (pure-dummy iter).
#   Result: in DP mode, the two counters diverge whenever the engine is
#   idle — making it impossible to map a captured nsys window back to a
#   specific Iteration(N) range from the engine log.
#
# This breaks the standard workflow where users read the engine log to
# decide which iteration range to capture (e.g., `start_step=6000` to land
# on iter 6000 of the bench's steady-state main loop), because in vLLM DP
# the wrapper.step() count != Iteration(N) count.
#
# The fix: when DPEngineCoreProc.run_busy_loop dispatches
# `execute_dummy_batch` (the pure-dummy iter path), also emit an
# `Iteration(N)` log entry with all-zero counts and increment the same
# `_iteration_index` counter that EngineCoreProc.log_iteration_details
# uses. Now Iteration(N) ticks once per iter for ALL iter types,
# matching wrapper.step().
#
# Gated on `enable_logging_iteration_details=True` so it's no-op when
# iter-logging is off (default). Idempotent. Affects DPEngineCoreProc
# only; non-DP `EngineCoreProc.run_busy_loop` is untouched because it
# doesn't have an execute_dummy_batch path (blocks on input_queue when
# no requests, so no pure-dummy iters happen).
#
# Important: KV-only iters (tokens=0 but has_requests=True) already go
# through EngineCore.step() and its log_iteration_details, which emits
# Iteration(N) and ticks _iteration_index. The patch must NOT
# double-emit on those. We use a delta check on _iteration_index
# (snapshot before _process_engine_step, compare after) as the signal —
# only emit a dummy log when the index did NOT advance during the step
# call, which is the exact upstream contract for "step() decided not to
# log this iter". This is more robust than inferring from `not
# executed` since that flag is True for BOTH KV-only and pure-dummy.
# The emit is tagged "[dp-idle]" right after the Iteration(N) prefix
# so users can `grep -v "\\[dp-idle\\]"` to exclude synthetic iters
# from throughput counts while keeping the index continuous for
# capture-window planning.
python3 - <<'PYPATCH_DUMMY_LOG_SYNC'
import pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[vllm-dummy-log-sync-patch] core.py not found, skipping")
    sys.exit(0)

content = target.read_text()

old_block = """            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()"""

new_block = """            # [PATCHED by vllm-container-deps-nsys.sh] Snapshot
            # _iteration_index BEFORE _process_engine_step so we can later
            # detect whether log_iteration_details inside EngineCore.step()
            # actually emitted an Iteration(N) log (i.e. scheduler had
            # requests). This lets us emit a dummy Iteration(N) log for
            # pure-dummy iters ONLY (no requests at all), without
            # double-counting KV-only iters where step() already emitted
            # with all-zero counts.
            _vlnsys_iter_log_enabled = (
                self.vllm_config.observability_config.enable_logging_iteration_details
            )
            if _vlnsys_iter_log_enabled:
                _vlnsys_iter_before = getattr(self, "_iteration_index", 0)

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # [PATCHED] Only emit dummy log if step() did NOT already
                # tick _iteration_index (i.e. this is a pure-dummy iter, not
                # a KV-only iter). Using the index delta as the signal is
                # more robust than inferring from `not executed`, which
                # covers both KV-only and pure-dummy cases. The emit is
                # tagged with "[dp-idle]" right after the Iteration(N)
                # prefix so it is grep-distinguishable from real engine
                # iters that also log all-zero counts on KV-transfer-only
                # iters — e.g. `grep "Iteration(" | grep -v "dp-idle"`
                # gives just the real-work iters for throughput stats.
                _vlnsys_emit_dummy_log = (
                    _vlnsys_iter_log_enabled
                    and getattr(self, "_iteration_index", 0) == _vlnsys_iter_before
                )
                if _vlnsys_emit_dummy_log:
                    _vlnsys_before = time.monotonic()
                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()
                if _vlnsys_emit_dummy_log:
                    _vlnsys_idx = getattr(self, "_iteration_index", 0)
                    logger.info(
                        "".join(
                            [
                                "Iteration(",
                                str(_vlnsys_idx),
                                ") [dp-idle]: 0 context requests, 0 context tokens, ",
                                "0 generation requests, 0 generation tokens, ",
                                "iteration elapsed time: ",
                                format((time.monotonic() - _vlnsys_before) * 1000, ".2f"),
                                " ms",
                            ]
                        )
                    )
                    self._iteration_index = _vlnsys_idx + 1"""

if new_block in content:
    print(f"[vllm-dummy-log-sync-patch] Already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    # Hard fail on source drift — see the matching block in PYPATCH_DUMMY_TICK.
    print("[vllm-dummy-log-sync-patch] ERROR: expected block not found in", target)
    print("[vllm-dummy-log-sync-patch] vLLM source has drifted past v0.21.0 — update PYPATCH_DUMMY_LOG_SYNC")
    print("[vllm-dummy-log-sync-patch] in configs/patches/vllm-container-deps-nsys.sh to match the new shape.")
    sys.exit(1)

content = content.replace(old_block, new_block)
target.write_text(content)
print(f"[vllm-dummy-log-sync-patch] Patched (added dummy-iter Iteration(N) emit) in {target}")
PYPATCH_DUMMY_LOG_SYNC

# ----------------------------------------------------------------------------
# Patch vLLM's CudaProfilerWrapper._start/_stop to put a CROSS-RANK barrier
# around the cudaProfilerStart/Stop CUPTI toggle.
#
# Root cause (confirmed by elimination — see VLLM_NSYS_DP_COUNTER_UNIFICATION.md
# "OPEN BUG"):
#   vLLM's nsys CUPTI capture toggle (cudaProfilerStart/Stop) is NOT
#   coordinated across DP ranks. Toggling CUPTI on a subset (leader-only)
#   or asynchronously (all-N ranks at uncoordinated instants) WHILE the
#   NVLS / NVSHMEM one-sided EP all2all is under full steady-state load
#   desyncs the collective state. The desync is latent and detonates at the
#   next small/uneven-batch collective — the benchmark drain tail — as a
#   synchronized collective deadlock (all ranks stall at the same iter;
#   surfaces as shm_broadcast RPC TimeoutError). The more asymmetric the
#   toggle, the sooner it fires:
#     - leader-only (1-of-16 captures): deadlock right after the capture
#       window (v39 #53855636, iter ~6910)
#     - all-16 fanout + local cuda.synchronize fence: survives to the
#       measured-run drain tail (v35/36/37, iter ~17000)
#     - capture at rampup / light load (v34, delay=500): clean (the toggle
#       perturbs a lightly-loaded collective)
#
# Fix: a real CROSS-RANK barrier on the DP gloo CPU group around BOTH the
# start and stop toggles, so the CUPTI capture window opens and closes as a
# synchronized global event across all DP ranks — no rank-to-rank desync.
# This is exactly SGLang's profile_manager.py shape
# (torch.distributed.barrier(dp_tp_cpu_group) around the profiler toggle),
# adapted to vLLM's per-rank profiler. The local torch.cuda.synchronize()
# fence is KEPT around stop (drains in-flight GPU work before teardown).
#
# REQUIRES per-rank profile fanout (every DP rank calls /engine/start_profile)
# so that ALL ranks reach the barrier — do NOT combine with
# PROFILE_DECODE_LEADER_ONLY (only 1 rank would call the barrier -> hang).
# The barrier is a no-op when torch.distributed is not initialized or the DP
# group is unavailable (single-rank / non-DP), so TP/TEP captures are safe.
#
# Idempotent. Two edits: (1) inject the _vlnsys_dp_barrier helper after the
# module logger; (2) wrap _start/_stop.
python3 - <<'PYPATCH_STOP_FENCE'
import pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/profiler/wrapper.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/profiler/wrapper.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[vllm-stop-fence-patch] vllm/profiler/wrapper.py not found, skipping")
    sys.exit(0)

content = target.read_text()

# --- Edit 1: inject the cross-rank barrier helper after the module logger ---
helper_anchor = "logger = init_logger(__name__)"
helper_def = '''logger = init_logger(__name__)


def _vlnsys_dp_barrier() -> None:
    # [PATCHED by vllm-container-deps-nsys.sh] Cross-rank CPU barrier on the
    # DP gloo group so the CUPTI capture toggle (cudaProfilerStart/Stop) is a
    # synchronized global event across all DP ranks. Without it an
    # uncoordinated toggle under full NVLS/NVSHMEM collective load desyncs
    # ranks and deadlocks at the next small/uneven-batch collective (the
    # benchmark drain tail). No-op if distributed is not initialized or the
    # DP group / cpu_group is unavailable (single-rank / non-DP). Requires
    # ALL participating ranks to call it -> use only with per-rank profile
    # fanout, never leader-only.
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return
        from vllm.distributed.parallel_state import get_dp_group

        cpu_group = getattr(get_dp_group(), "cpu_group", None)
        if cpu_group is None:
            return
        dist.barrier(group=cpu_group)
    except Exception as exc:  # never let the barrier kill the run
        logger.warning("[vllm-stop-fence-patch] DP barrier skipped: %s", exc)'''

if "_vlnsys_dp_barrier" not in content:
    if helper_anchor not in content:
        print("[vllm-stop-fence-patch] ERROR: logger anchor not found in", target)
        print("[vllm-stop-fence-patch] vLLM source has drifted past v0.21.0 — update PYPATCH_STOP_FENCE")
        sys.exit(1)
    content = content.replace(helper_anchor, helper_def, 1)

# --- Edit 2: wrap _start/_stop with the barrier (+ keep the local fence) ---
old_block = """    @override
    def _start(self) -> None:
        self._cuda_profiler.start()

    @override
    def _stop(self) -> None:
        self._cuda_profiler.stop()"""

new_block = """    @override
    def _start(self) -> None:
        # [PATCHED] cross-rank barrier so all DP ranks open the CUPTI capture
        # window together (uncoordinated cudaProfilerStart under collective
        # load desyncs NVLS/NVSHMEM -> drain-tail deadlock).
        _vlnsys_dp_barrier()
        self._cuda_profiler.start()
        _vlnsys_dp_barrier()

    @override
    def _stop(self) -> None:
        # [PATCHED] symmetric cross-rank barrier + local GPU drain around
        # cudaProfilerStop. Matches SGLang profile_manager's
        # torch.distributed.barrier(dp_tp_cpu_group) around the profiler stop.
        _vlnsys_dp_barrier()
        torch.cuda.synchronize()
        self._cuda_profiler.stop()
        torch.cuda.synchronize()
        _vlnsys_dp_barrier()"""

if new_block in content:
    print(f"[vllm-stop-fence-patch] Already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    # Hard fail on source drift — see the matching block in PYPATCH_DUMMY_TICK.
    print("[vllm-stop-fence-patch] ERROR: expected _start/_stop block not found in", target)
    print("[vllm-stop-fence-patch] vLLM source has drifted past v0.21.0 — update PYPATCH_STOP_FENCE")
    print("[vllm-stop-fence-patch] in configs/patches/vllm-container-deps-nsys.sh to match the new shape.")
    sys.exit(1)

content = content.replace(old_block, new_block)
target.write_text(content)
print(f"[vllm-stop-fence-patch] Patched (cross-rank DP barrier + cuda.synchronize around cudaProfilerStart/Stop) in {target}")
PYPATCH_STOP_FENCE
