#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same as vllm-container-deps.sh, plus a STARTUP low-concurrency decode warmup
# ramp injected into vLLM's gpu_worker.compile_or_warm_up_model(), right after
# cudagraph capture.
#
# Why: the Kimi-K2.5 8k1k multi-sweep "warm" server reaches c=32 at ~16.8 ms
# TPOT only AFTER serving low-concurrency phases (c=1..16); a fresh single-conc
# c=32 server stays at ~20.9 ms, and repeating c=32 (c32x3) does NOT warm it —
# the warming is specifically from the low-conc ramp (prefix-ablation: c=16
# alone ≈75%). Operationally pre-warming with real low-conc traffic isn't
# viable in production, so this replays decode-shaped dummy forwards at small
# batch sizes at startup to try to establish the same state without traffic.
#
# This is also a TEST of the mechanism: if dummy passes reproduce the warming
# (cold c=32 drops toward ~16.8), the warmed state is something a dummy forward
# can establish (allocator/cache/layout); if not, real-serving state (growing
# KV / real routing) is required.
#
# Gated on VLLM_WARMUP_RAMP_ITERS (0 = no-op). Sizes via VLLM_WARMUP_RAMP_SIZES
# (default "1,2,4,8,16"). No effect under enforce-eager (no cudagraph regime).
set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl
pip install msgpack
if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

python3 - <<'PYPATCH_WARMUP_RAMP'
import pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_worker.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[warmup-ramp] gpu_worker.py not found, skipping")
    sys.exit(0)

src = target.read_text()
if "[warmup-ramp]" in src:
    print("[warmup-ramp] already patched:", target)
    sys.exit(0)

anchor = (
    "        cuda_graph_memory_bytes = 0\n"
    "        if not self.model_config.enforce_eager:\n"
    "            cuda_graph_memory_bytes = self.model_runner.capture_model()\n"
)
if anchor not in src:
    print("[warmup-ramp] ERROR: capture_model anchor not found in", target)
    print("[warmup-ramp] vLLM gpu_worker has drifted past v0.22.0 — update PYPATCH_WARMUP_RAMP")
    sys.exit(1)

inject = anchor + (
    "\n"
    "        # [warmup-ramp] sustained low-concurrency decode warmup (srt-slurm patch).\n"
    "        # Replays decode-shaped dummy forwards at small batch sizes to establish\n"
    "        # the state the multi-sweep low-conc ramp gives a warm server, without\n"
    "        # real low-conc traffic. Gated on VLLM_WARMUP_RAMP_ITERS (0 = off).\n"
    "        import os as _wr_os\n"
    "        _wr_iters = int(_wr_os.environ.get(\"VLLM_WARMUP_RAMP_ITERS\", \"0\"))\n"
    "        if _wr_iters > 0 and not self.model_config.enforce_eager:\n"
    "            _wr_sizes = [int(x) for x in _wr_os.environ.get(\"VLLM_WARMUP_RAMP_SIZES\", \"1,2,4,8,16\").split(\",\") if x.strip()]\n"
    "            logger.info(\"[warmup-ramp] sustained low-conc decode warmup: %d iters x sizes %s\", _wr_iters, _wr_sizes)\n"
    "            for _wr_i in range(_wr_iters):\n"
    "                for _wr_bs in _wr_sizes:\n"
    "                    try:\n"
    "                        self.model_runner._dummy_run(_wr_bs, uniform_decode=True, skip_eplb=True, remove_lora=False)\n"
    "                    except Exception as _wr_e:\n"
    "                        logger.warning(\"[warmup-ramp] dummy_run(bs=%d) failed: %s\", _wr_bs, _wr_e)\n"
    "                        _wr_iters = 0\n"
    "                        break\n"
    "                if _wr_iters == 0:\n"
    "                    break\n"
    "            import torch as _wr_torch\n"
    "            _wr_torch.cuda.synchronize()\n"
    "            logger.info(\"[warmup-ramp] done\")\n"
)

target.write_text(src.replace(anchor, inject, 1))
print("[warmup-ramp] patched:", target)
PYPATCH_WARMUP_RAMP

echo "[warmup-ramp] setup script done"
