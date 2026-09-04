#!/usr/bin/env bash
# Wei's GB300 stack plus one change: let the KDA ssm state follow the dtype knob.
# =============================================================================
# WHAT AND WHY. `MambaStateDtypeCalculator.kda_state_dtype()` returns
# `(state_dtype, torch.float32)` -- the conv half already follows
# `--mamba-cache-dtype` and resolves to bf16, the ssm half is hardcoded. The
# knob it should read, `--mamba-ssm-cache-dtype`, already exists in CacheConfig
# (`auto|float32|float16|bfloat16`); nothing was wired to it on the KDA path.
#
# THE SIZE OF IT, from our own c70 worker log:
#
#   Setting attention block size to 1536 tokens to ensure that attention page
#   size is >= mamba page size.
#   Padding mamba page size by 8.68% to ensure that mamba page size and
#   attention page size are exactly equal.
#
# One KDA page is as large as 1536 tokens of MLA KV at fp8. So this is not only
# a capacity change -- capacity is measured and worth nothing here, gmu
# 0.92 -> 0.94 grew the pool 12-38% for 0% -- it lets the attention block size
# come down, which moves prefix-cache granularity, and coverage is the only
# thing that moves y on this workload.
#
# `auto` STILL MEANS FLOAT32. Mamba-2's `_mamba_state_dtype` makes `auto` follow
# the conv dtype; copying that here would move every K3 run to a bf16 recurrent
# state without anyone asking. The ssm state is recurrent, so error accumulates
# per token rather than being a one-shot cast. The arm sets the dtype
# explicitly and the control is the same script without it.
# =============================================================================
set -euo pipefail

BASE=/configs/patches/k3-wei-infmax-3696c772-on-46638857.patch
KDA=/configs/patches/k3-kda-ssm-dtype.patch
SITE=$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin)))")

echo "=== wei-gb300-kdassm: applying $(basename "$BASE") then $(basename "$KDA") ==="

python3 - <<'PY'
import vllm, sys
want = "46638857"
got = vllm.__version__
if want not in got and "3696c77" not in got:
    sys.exit(f"wei-gb300-kdassm: refusing, vllm is {got}, expected a build of "
             f"46638857 (or Wei's 3696c77). Applying a 6,368-line patch to the "
             f"wrong tree fuzzes 73 files silently.")
print(f"    vllm {got}")
PY

cd "$SITE"
patch -p1 --forward --batch < "$BASE" >/dev/null && echo "    base patch applied"
patch -p1 --forward --batch < "$KDA" >/dev/null && echo "    kda ssm dtype patch applied"

# Prove the knob is wired, on the tree that will actually serve. A grep would
# pass on a comment; this calls the function.
python3 - <<'PY'
import sys, torch
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C
auto = C.kda_state_dtype(torch.bfloat16, "auto", "auto")
bf16 = C.kda_state_dtype(torch.bfloat16, "auto", "bfloat16")
fp32 = C.kda_state_dtype(torch.bfloat16, "auto", "float32")
print(f"    kda_state_dtype auto     -> {auto}")
print(f"    kda_state_dtype bfloat16 -> {bf16}")
print(f"    kda_state_dtype float32  -> {fp32}")
if auto[1] is not torch.float32:
    sys.exit("wei-gb300-kdassm: 'auto' no longer means float32 -- the default moved, "
             "which would change the control as well as the arm.")
if bf16[1] is not torch.bfloat16:
    sys.exit("wei-gb300-kdassm: the knob is not wired; ssm state stayed "
             f"{bf16[1]} when asked for bfloat16.")
print("    verified: auto keeps fp32, explicit bfloat16 takes effect")
PY

# The five modules a K3 worker loads, same as the base script.
python3 - <<'PY'
import importlib, sys
mods = ["vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker",
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker",
        "vllm.models.kimi_k3.nvidia.kda",
        "vllm.v1.attention.backends.mla.tokenspeed_mla",
        "vllm.v1.worker.gpu.spec_decode.dspark.utils"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"    import {m.split('.')[-1]:<30} ok")
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
if bad:
    sys.exit("wei-gb300-kdassm: import check failed\n  " + "\n  ".join(bad))
PY

echo "=== wei-gb300-kdassm: done ==="
