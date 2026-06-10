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

# ----------------------------------------------------------------------------
# Patch FlashInfer's AutoTuner.choose_one profiling loop to log EACH candidate
# tactic's measured time (and inf on profiling failure) for the
# trtllm_fp4_block_scale_moe op. Gated on VLLM_DUMP_AUTOTUNE_TIMES=1. Read-only
# (adds logging only). This distinguishes WHY the chosen tactic differs across
# launches: (a) near-tied successful times reordered by noise, (b) the fast
# tactic profiling-FAILED (t_ms=inf) — cf. FlashInfer #3168 autotune device-state
# corruption, or (c) the startup profile mis-ranks the fast tactic (measures it
# slower) for the synthetic m=32 shape. Idempotent; hard-fails on source drift.
python3 - <<'PYPATCH_AUTOTUNE_TIMES'
import pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/autotuner.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/flashinfer/autotuner.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[autotune-times] flashinfer/autotuner.py not found, skipping")
    sys.exit(0)

content = target.read_text()

# Insert a per-candidate timing log right before the argmin comparison.
old_block = """                                if time_measured < min_time:
                                    min_time = time_measured
                                    runner_id, tactic = r_id, tac"""

new_block = """                                import os as _att_os
                                if _att_os.environ.get("VLLM_DUMP_AUTOTUNE_TIMES") and "trtllm_fp4_block_scale_moe" in str(custom_op):
                                    try:
                                        logger.info(
                                            "[autotune-times] op=%s shapes=%s r=%d tac=%s t_ms=%s",
                                            custom_op, self._get_input_sizes(tensors), r_id, tac, time_measured,
                                        )
                                    except Exception:
                                        pass
                                if time_measured < min_time:
                                    min_time = time_measured
                                    runner_id, tactic = r_id, tac"""

if "[autotune-times]" in content:
    print(f"[autotune-times] already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    print("[autotune-times] ERROR: expected argmin block not found in", target)
    print("[autotune-times] FlashInfer autotuner.py has drifted — update PYPATCH_AUTOTUNE_TIMES")
    sys.exit(1)

content = content.replace(old_block, new_block, 1)
target.write_text(content)
print(f"[autotune-times] Patched (per-candidate timing log) in {target}")
PYPATCH_AUTOTUNE_TIMES

# ----------------------------------------------------------------------------
# Patch FlashInfer AutoTuner.choose_one SERVING path (is_tuning_mode False) to
# log the (runner, tactic) the MoE op actually dispatches per shape at serving
# time — deduped per (shape, runner, tactic) to bound volume. Gated on
# VLLM_DUMP_AUTOTUNE_DISPATCH=1. Read-only. This pins whether the serving
# dispatch uses the startup-cached config and whether it differs warm vs cold
# (complements nsys, which shows the executed kernel NAME). Idempotent.
python3 - <<'PYPATCH_AUTOTUNE_DISPATCH'
import pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/autotuner.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/flashinfer/autotuner.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[autotune-dispatch] flashinfer/autotuner.py not found, skipping")
    sys.exit(0)

content = target.read_text()

old_block = """                return runner, tactic

            assert len(runners) > 0, \"At least one runner is required\""""

new_block = """                import os as _disp_os
                if _disp_os.environ.get(\"VLLM_DUMP_AUTOTUNE_DISPATCH\") and \"trtllm_fp4_block_scale_moe\" in str(custom_op):
                    try:
                        _disp_seen = getattr(self, \"_vlnsys_disp_seen\", None)
                        if _disp_seen is None:
                            _disp_seen = set(); self._vlnsys_disp_seen = _disp_seen
                        _disp_key = (str(input_shapes), runner_id, str(tactic))
                        if _disp_key not in _disp_seen:
                            _disp_seen.add(_disp_key)
                            logger.info(\"[autotune-dispatch] op=%s shapes=%s runner=%s tactic=%s cache_hit=%s\",
                                        custom_op, input_shapes, runner_id, tactic, is_cache_hit)
                    except Exception:
                        pass
                return runner, tactic

            assert len(runners) > 0, \"At least one runner is required\""""

if "[autotune-dispatch]" in content:
    print(f"[autotune-dispatch] already patched, skipping. File: {target}")
    sys.exit(0)

if old_block not in content:
    print("[autotune-dispatch] ERROR: expected serving-return block not found in", target)
    print("[autotune-dispatch] FlashInfer autotuner.py has drifted — update PYPATCH_AUTOTUNE_DISPATCH")
    sys.exit(1)

content = content.replace(old_block, new_block, 1)
target.write_text(content)
print(f"[autotune-dispatch] Patched (serving-time dispatch log) in {target}")
PYPATCH_AUTOTUNE_DISPATCH
