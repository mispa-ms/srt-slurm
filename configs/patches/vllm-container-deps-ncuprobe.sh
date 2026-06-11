#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# STANDALONE ncu microbench of the Kimi-K2.5 NVFP4 MoE GEMM at the decode shape
# (m=32). In-situ ncu on the live TP=4 server fails (UnknownError) because the
# MoE GEMM is collective-coupled (NCCL allreduce) and ncu's per-kernel replay
# desyncs the ranks. This probe runs the SAME op in a SINGLE process on ONE GPU
# (no TP, no collective), so ncu profiles it cleanly — giving the kernel's
# intrinsic L2 hit-rate / DRAM bytes / SM throughput (the isolated/cold regime,
# which we measured ≈ cold serving). Combined with the nsys in-situ DRAM-read
# delta (cold 51 vs warm 32), this tests whether the kernel is memory-bound.
#
# Runs in the container setup step (GPU node). Guarded by nvidia-smi: if no GPU
# is visible at setup time the probe is skipped and that fact is logged, so a
# single run also tells us whether setup has GPU access.
set -uo pipefail   # NOT -e: probe failure must not abort the job

apt-get -y update && apt-get install -y --no-install-recommends --allow-change-held-packages numactl
pip install msgpack
if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

# ncu install (nsight-compute is a virtual package -> pick a concrete version)
if ! command -v ncu >/dev/null 2>&1; then
    apt-get -y update
    apt-get install -y --no-install-recommends nsight-compute-2025.3.0 || \
        apt-get install -y --no-install-recommends nsight-compute-2025.1.1 || \
        apt-get install -y --no-install-recommends nsight-compute-2024.3.2 || true
fi
if ! command -v ncu >/dev/null 2>&1; then
    _ncu_bin="$(find /opt/nvidia/nsight-compute -maxdepth 2 -name ncu -type f 2>/dev/null | sort | tail -1)"
    [ -n "${_ncu_bin}" ] && ln -sf "${_ncu_bin}" /usr/local/bin/ncu
fi

echo "[ncuprobe] ===== nvidia-smi at setup (does setup have GPU?) ====="
nvidia-smi -L 2>&1 | head -8 || echo "[ncuprobe] nvidia-smi FAILED — no GPU at setup"
ncu --version 2>&1 | head -3 || echo "[ncuprobe] ncu NOT available"

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[ncuprobe] No GPU at setup time — skipping probe (will need a serve-time hook instead)."
    exit 0
fi

# ---- write the standalone probe (single process, 1 GPU, no TP/collective) ----
cat > /tmp/moe_l2_probe.py <<'PROBE'
import torch
from flashinfer import fp4_quantize
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.fused_moe.core import RoutingMethodType
try:
    from flashinfer.fused_moe.core import ActivationType
    SWIGLU = ActivationType.Swiglu.value
except Exception:
    SWIGLU = 1
try:
    from flashinfer.autotuner import autotune
except Exception:
    autotune = None

# Kimi-K2.5 NVFP4 MoE decode shape (per the 8k1k AGG TP=4 config; here single GPU,
# local_num_experts = full num_experts since no TP).
M = 32                 # decode batch (num_tokens)
E = 384                # num_experts
H = 7168               # hidden_size
I = 2048               # intermediate_size
TOPK = 8
dev = torch.device("cuda:0")

w13 = torch.randn(E, I * 2, H, device=dev).to(torch.bfloat16)
w2 = torch.randn(E, H, I, device=dev).to(torch.bfloat16)
b13 = torch.randn(E, I * 2, device=dev) * 10
b2 = torch.randn(E, H, device=dev) * 10
amax = torch.tensor([448.0 * 6.0], device=dev)
w13, w13s = fp4_quantize(w13, amax, sf_vec_size=16, sf_use_ue8m0=False)
w13s = w13s.view(torch.float8_e4m3fn).reshape(E, I * 2, -1)
w2, w2s = fp4_quantize(w2, amax, sf_vec_size=16, sf_use_ue8m0=False)
w2s = w2s.view(torch.float8_e4m3fn).reshape(E, H, -1)
gsc = 1.0 / 448.0 / 6.0
o1 = torch.tensor([gsc * gsc] * E, device=dev)
o2 = torch.tensor([gsc * gsc] * E, device=dev)

hs = torch.randn(M, H, device=dev).to(torch.bfloat16)
hs, hss = fp4_quantize(hs, amax, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=False)
hss = hss.view(torch.float8_e4m3fn).reshape(M, -1)
rl = torch.rand(M, E, device=dev).to(torch.bfloat16)

kw = dict(
    routing_logits=rl, routing_bias=None,
    hidden_states=hs, hidden_states_scale=hss,
    gemm1_weights=w13, gemm1_weights_scale=w13s, gemm1_bias=b13,
    gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
    gemm2_weights=w2, gemm2_weights_scale=w2s, gemm2_bias=b2,
    output1_scale_scalar=o1, output1_scale_gate_scalar=o1, output2_scale_scalar=o2,
    num_experts=E, top_k=TOPK, n_group=None, topk_group=None,
    intermediate_size=I, local_expert_offset=0, local_num_experts=E,
    routed_scaling_factor=None, routing_method_type=RoutingMethodType.Renormalize.value,
    do_finalize=True, activation_type=SWIGLU, tune_max_num_tokens=8192,
)

# Autotune (so the profiled kernel is the tactic serving would pick at m=32).
if autotune is not None:
    with autotune(True):
        for _ in range(3):
            trtllm_fp4_block_scale_moe(**kw)
torch.cuda.synchronize()

# Steady loop — ncu skips warmup launches and profiles a few of these.
for _ in range(60):
    trtllm_fp4_block_scale_moe(**kw)
torch.cuda.synchronize()
print("[ncuprobe] probe loop done", flush=True)
PROBE

echo "[ncuprobe] ===== running ncu on the MoE GEMM (single process) ====="
export TORCH_CUDA_ARCH_LIST="10.0"
# Profile the gate/up u2 GEMM. cache-control none = no L2 flush (serving-like).
# Minimal metrics to stay ~single-pass. launch-skip past autotune+warmup.
ncu --kernel-name "regex:bmm.*E2m1.*u2" \
    --launch-skip 40 --launch-count 4 \
    --cache-control none \
    --metrics lts__t_sector_hit_rate.pct,dram__bytes_read.sum,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    --csv --log-file /logs/moe_l2_probe.csv -f \
    python3 /tmp/moe_l2_probe.py 2>&1 | tee /logs/moe_l2_probe.stdout | tail -40

echo "[ncuprobe] ===== ncu CSV ====="
cat /logs/moe_l2_probe.csv 2>/dev/null || echo "[ncuprobe] no CSV produced"
echo "[ncuprobe] done"
