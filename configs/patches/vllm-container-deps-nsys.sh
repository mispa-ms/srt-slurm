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

# Patch vLLM's CudaProfilerWrapper to call torch.cuda.synchronize() before
# cudaProfilerStart/Stop. Without this, the auto-stop path (max_iterations
# reached -> wrapper.step() -> _call_stop()) fires cudaProfilerStop while
# kernels are still in flight inside the FULL_DECODE_ONLY cudagraph that
# vLLM replays on every decode step. CUPTI's stop interaction with an
# in-flight graph leaves the CUDA context in a broken state — a few seconds
# later the next sample_tokens.async_copy_ready_event.synchronize() returns
# cudaErrorLaunchFailure on the profiled rank, the engine_cores cascade
# into TimeoutError on RPC, and decode dies even though the nsys file has
# already flushed cleanly.
#
# Pre-stop sync forces all queued work to drain before the profile boundary,
# matching SGLang's profile_manager behavior (which doesn't hit this bug at
# the same trace volume).
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

if "torch.cuda.synchronize()" in content:
    print("[vllm-cuda-profiler-sync-patch] Already patched, skipping.")
    sys.exit(0)
if old_start not in content or old_stop not in content:
    print("[vllm-cuda-profiler-sync-patch] Expected blocks not found — vLLM source drift?")
    print("[vllm-cuda-profiler-sync-patch] File:", target)
    sys.exit(0)

content = content.replace(old_start, new_start).replace(old_stop, new_stop)
target.write_text(content)
print("[vllm-cuda-profiler-sync-patch] Patched CudaProfilerWrapper._start/_stop in", target)
PYPATCH
