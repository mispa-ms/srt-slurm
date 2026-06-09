#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same as vllm-container-deps.sh, plus a READ-ONLY dump of the FlashInfer
# autotuner's chosen kernel tactic per shape, right after vLLM's startup
# autotune pass. Used to diagnose the Kimi-K2.5 NVFP4 / B300 8k1k multi-sweep
# vs single-conc TPOT gap: the warm leg runs the fast MoE GEMM tactic and the
# cold leg runs an ~4-8x slower one (`u2`) for the c=32 decode shape. This
# patch logs which tactic each server's startup autotune actually selected for
# `trtllm_fp4_block_scale_moe`, so a warm vs cold launch can be compared
# directly (tests autotune nondeterminism-across-launches).
#
# The patch does NOT change tactic selection — it only reads
# AutoTuner.get().profiling_cache after the existing autotune() context and
# logs matching entries to the server stdout (out.log, always archived).
# Persistent cache stays OFF (vLLM keeps it off for #2861/#3367 collisions);
# we do not touch that.
#
# Gated on VLLM_DUMP_AUTOTUNE_CACHE=1 so it is a no-op unless explicitly set.

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# ----------------------------------------------------------------------------
# Patch vLLM's flashinfer_autotune() to log the autotuner's chosen tactic per
# shape after the startup autotune pass. Read-only; gated on
# VLLM_DUMP_AUTOTUNE_CACHE. Idempotent via string-presence check. Hard-fails on
# source drift so a newer vLLM image forces a patch update rather than silently
# no-op'ing.
python3 - <<'PYPATCH_AUTOTUNE_DUMP'
import pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/model_executor/warmup/kernel_warmup.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[vllm-autotune-dump] kernel_warmup.py not found, skipping")
    sys.exit(0)

content = target.read_text()

old_block = """    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune():
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return"""

new_block = """    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune():
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        # [PATCHED by vllm-container-deps-autotunedump.sh] Read-only dump of the
        # autotuner's chosen tactic per shape (gated on VLLM_DUMP_AUTOTUNE_CACHE).
        # Does NOT change selection; logs to stdout for warm-vs-cold comparison.
        import os as _adump_os
        if _adump_os.environ.get("VLLM_DUMP_AUTOTUNE_CACHE"):
            try:
                from flashinfer.autotuner import AutoTuner as _ADTuner
                _adump_rank = get_world_group().rank_in_group
                _adump_cache = getattr(_ADTuner.get(), "profiling_cache", {})
                _adump_n = 0
                for _adump_k, _adump_v in _adump_cache.items():
                    _adump_ks = str(_adump_k)
                    _adump_runner = _adump_v[0] if isinstance(_adump_v, (list, tuple)) and len(_adump_v) > 0 else None
                    _adump_tactic = _adump_v[1] if isinstance(_adump_v, (list, tuple)) and len(_adump_v) > 1 else None
                    logger.info(
                        "[autotune-dump] rank=%d runner=%s tactic=%s key=%s",
                        _adump_rank, _adump_runner, _adump_tactic, _adump_ks,
                    )
                    _adump_n += 1
                logger.info("[autotune-dump] rank=%d dumped %d profiling_cache entries", _adump_rank, _adump_n)
            except Exception as _adump_e:
                logger.warning("[autotune-dump] failed: %r", _adump_e)
        get_world_group().barrier()
        return"""

if "[autotune-dump]" in content:
    print(f"[vllm-autotune-dump] already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    print("[vllm-autotune-dump] ERROR: expected flashinfer_autotune block not found in", target)
    print("[vllm-autotune-dump] vLLM source has drifted past v0.22.0 — update PYPATCH_AUTOTUNE_DUMP")
    print("[vllm-autotune-dump] in configs/patches/vllm-container-deps-autotunedump.sh to match the new shape.")
    sys.exit(1)

content = content.replace(old_block, new_block, 1)
target.write_text(content)
print(f"[vllm-autotune-dump] Patched (read-only autotune tactic dump) in {target}")
PYPATCH_AUTOTUNE_DUMP
