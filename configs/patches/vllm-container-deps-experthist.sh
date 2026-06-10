#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same as vllm-container-deps.sh, plus a READ-ONLY per-expert routing histogram
# dump for the FP4 fused-MoE path. Used to settle the Kimi-K2.5 NVFP4 / B300
# 8k1k multi-sweep vs single-conc TPOT gap: the same MoE GEMM kernel runs ~4-8x
# slower under the cold single-conc protocol (config / kernel-variant / grid /
# dispatch / host-NUMA all ruled out). Remaining hypotheses are on-device
# memory-bound: (a) L2 residency, (b) expert-routing spread (cold touching a
# different SET of experts -> more distinct expert weights streamed from DRAM).
# This dump tests (b): it captures the kernel's selected expert IDs per token
# and logs the per-expert routing distribution, so a cold vs warm run can be
# compared directly. If the distributions match -> routing is NOT the cause
# (supports pure L2 residency); if cold is more spread -> routing matters.
#
# Mechanism: flashinfer's trtllm_fp4_block_scale_moe() accepts an optional
# `routing_replay_out` [num_tokens, top_k] int16 buffer that the kernel fills
# with the selected (global) expert IDs ("routing replay"). vLLM passes None,
# so we monkeypatch the public wrapper to allocate it and bincount the result.
# The histogram is accumulated ON GPU (no per-call sync) and only copied to host
# + logged every VLLM_DUMP_EXPERT_HIST_EVERY calls. Output (= expected MoE math
# is unchanged; routing_replay_out is a pure side output).
#
# IMPORTANT: the MoE python wrapper only runs in eager mode. Under CUDA graphs
# the wrapper executes during capture only (not replay), where stream capture is
# active and this patch deliberately no-ops. So the dump run MUST set
# enforce-eager: true. Routing is a function of token content, independent of
# eager vs graph, so the routing distribution measured in eager is the same one
# the graphed serving path sees.
#
# Gated on VLLM_DUMP_EXPERT_HIST=1 so it is a no-op unless explicitly set.
#   VLLM_DUMP_EXPERT_HIST_EVERY   (default 500) calls between host log lines
#   VLLM_DUMP_EXPERT_HIST_MAXTOK  (default 512) skip prefill (num_tokens>maxtok)

set -euo pipefail

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl

pip install msgpack

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# ----------------------------------------------------------------------------
# Append a monkeypatch to flashinfer's fused_moe/core.py that wraps the public
# trtllm_fp4_block_scale_moe() to inject routing_replay_out and accumulate a
# per-expert histogram. Appending (rather than editing the 40-line return block)
# is robust to minor source drift. Idempotent via marker check. Hard-fails if
# the function symbol is missing so a drifted image forces a patch update.
python3 - <<'PYPATCH_EXPERT_HIST'
import pathlib, sys

candidates = [
    pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py"),
    pathlib.Path("/usr/local/lib/python3.10/dist-packages/flashinfer/fused_moe/core.py"),
]
target = next((p for p in candidates if p.exists()), None)
if target is None:
    print("[expert-hist] flashinfer/fused_moe/core.py not found, skipping")
    sys.exit(0)

src = target.read_text()

MARKER = "_eh_orig_trtllm_fp4_block_scale_moe"
if MARKER in src:
    print("[expert-hist] already patched:", target)
    sys.exit(0)

# Anchor: the public wrapper must exist (FromLogits variant vLLM-NVFP4 calls).
if "def trtllm_fp4_block_scale_moe(" not in src:
    print("[expert-hist] ERROR: trtllm_fp4_block_scale_moe not found in", target)
    print("[expert-hist] FlashInfer source has drifted — update PYPATCH_EXPERT_HIST")
    sys.exit(1)

BLOCK = '''

# === expert-hist routing dump (VLLM_DUMP_EXPERT_HIST) — srt-slurm patch ===
# Wraps the public trtllm_fp4_block_scale_moe() to capture the kernel's selected
# expert IDs (routing_replay_out) and log the per-expert routing histogram.
# No-op unless VLLM_DUMP_EXPERT_HIST is set; skips during CUDA graph capture.
import os as _eh_os
import torch as _eh_torch
import numpy as _eh_np

_EH_STATE = {}

def _eh_accumulate(replay, num_experts, num_tokens, every):
    # replay: [rows>=num_tokens, top_k] int16 of selected (global) expert IDs.
    ids = replay[:num_tokens].reshape(-1).to(_eh_torch.int64)
    valid = (ids >= 0) & (ids < num_experts)
    counts = _eh_torch.bincount(ids[valid], minlength=num_experts)  # GPU, async
    st = _EH_STATE.get(num_tokens)
    if st is None:
        st = {"cum": _eh_torch.zeros(num_experts, dtype=_eh_torch.int64,
                                     device=counts.device), "n": 0}
        _EH_STATE[num_tokens] = st
    st["cum"] += counts
    st["n"] += 1
    if st["n"] % every == 0:
        cum = st["cum"].detach().cpu().numpy().astype(_eh_np.int64)
        s = int(cum.sum())
        nzc = int((cum > 0).sum())
        if s > 0:
            p = cum[cum > 0] / s
            ent = float(-(p * _eh_np.log2(p)).sum())
        else:
            ent = 0.0
        order = _eh_np.argsort(cum)[::-1][:10]
        top = [(int(e), int(cum[e])) for e in order]
        print("[expert-hist-cum] ntok=%d calls=%d distinct=%d/%d "
              "entropy=%.3f/%.2fbits total_slots=%d top10=%s"
              % (num_tokens, st["n"], nzc, num_experts, ent,
                 float(_eh_np.log2(num_experts)), s, top), flush=True)

_eh_orig_trtllm_fp4_block_scale_moe = trtllm_fp4_block_scale_moe

def trtllm_fp4_block_scale_moe(*args, **kwargs):
    _on = _eh_os.environ.get("VLLM_DUMP_EXPERT_HIST")
    _rl = kwargs.get("routing_logits")
    if _rl is None and args:
        _rl = args[0]
    _tk = kwargs.get("top_k")
    _alloc = False
    if _on and _rl is not None and _tk is not None \\
            and kwargs.get("routing_replay_out") is None \\
            and not _eh_torch.cuda.is_current_stream_capturing():
        try:
            _nt = int(_rl.shape[0])
            _maxt = int(_eh_os.environ.get("VLLM_DUMP_EXPERT_HIST_MAXTOK", "512"))
            if _nt <= _maxt:
                kwargs["routing_replay_out"] = _eh_torch.empty(
                    (_nt, int(_tk)), dtype=_eh_torch.int16, device=_rl.device)
                _alloc = True
        except Exception:
            _alloc = False
    ret = _eh_orig_trtllm_fp4_block_scale_moe(*args, **kwargs)
    if _alloc:
        try:
            _ne = kwargs.get("num_experts")
            _every = int(_eh_os.environ.get("VLLM_DUMP_EXPERT_HIST_EVERY", "500"))
            _eh_accumulate(kwargs["routing_replay_out"], int(_ne),
                           int(_rl.shape[0]), _every)
        except Exception:
            pass
    return ret
# === end expert-hist routing dump ===
'''

target.write_text(src + BLOCK)
print("[expert-hist] patched:", target)
PYPATCH_EXPERT_HIST

echo "[expert-hist] setup script done"
