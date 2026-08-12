#!/usr/bin/env bash
# k3-merged-v4 with vllm#51739 reverted, plus Mooncake under DCP.
#
# WHY THIS IMAGE EXISTS. v4 lost 6.1-10.1% tok/s/GPU against the v3 Mooncake
# ladder at c48-c70 once both sides ran the same prefill budget. The first
# hypothesis -- vllm#51726 moving _max_num_batched_tokens from 8192 to 16384 on
# >=160 GB GPUs, which these configs never named -- was tested and refuted:
# holding v4 at 8192 made it worse, so that default was masking part of the gap
# rather than causing it.
#
# What the runs point at instead is the step itself. At c70 and a matched budget
# ITL p90 is 144.4 ms on v3 and 157.4 ms on v4, with GPU prefix hit, ISL and
# error count unchanged. vllm#51739 (c76a425278) is the only one of the 68
# commits that changes compiled code -- 558 lines of cache_kernels.cu -- and it
# rewrote the long-context MLA gather to be page-based. Under DCP with
# kv-cache-dtype fp8 our call site takes exactly the branch it rewrote.
#
# WHY THIS COULD NOT BE A RUNTIME PATCH. #51739 changes no torch binding
# signature; ops.h and torch_bindings.cpp are untouched and only the .cu moved.
# The kernel body is in the .so, so reverting it is a rebuild.
#
# WHAT THE MARKER CHECKS, AND WHAT IT DOES NOT. The revert is nearly invisible
# from Python: exactly two call sites go back from `dst=workspace[:toks]` to
# `dst=workspace`, and in _context_parallel_compute_prefill_context the one that
# moved is ops.cp_gather_cache. That is the marker.
#
# It is deliberately NOT the branch this workload runs. With kv-cache-dtype fp8
# is_quantized_kv_cache() is true, so we call ops.gather_and_maybe_dequant_cache,
# whose call site #51739 left alone while rewriting the kernel underneath it.
# There is therefore no Python-visible marker on our own branch, and the
# neighbouring call in the same function is the closest honest proxy: it moves if
# and only if the commit is present. Without it a mispinned v4 would run green
# and be scored as the revert, which is the entire experiment.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-merged-v4-mooncake.sh"

python3 - <<'PY'
import ast
import pathlib

import vllm

path = pathlib.Path(vllm.__file__).parent / "model_executor/layers/attention/mla_attention.py"
fn = next(n for n in ast.walk(ast.parse(path.read_text()))
          if isinstance(n, ast.FunctionDef)
          and n.name == "_context_parallel_compute_prefill_context")

dst = {}
for n in ast.walk(fn):
    if isinstance(n, ast.Call):
        for k in n.keywords:
            if k.arg == "dst":
                dst[ast.unparse(n.func).split(".")[-1]] = ast.unparse(k.value)

assert "cp_gather_cache" in dst, (
    f"cp_gather_cache is no longer called in the DCP prefill context loop "
    f"(saw {sorted(dst)}); this marker no longer identifies the image"
)
assert dst["cp_gather_cache"] == "workspace", (
    f"wrong image: expected k3-merged-v4-rev51739, but cp_gather_cache passes "
    f"dst={dst['cp_gather_cache']}. vllm#51739 introduced that slice, so this is "
    f"an unreverted v4 image."
)
# The branch this workload actually runs. Its call site is unchanged by #51739 --
# only the kernel behind it was rewritten -- so this asserts presence, not form.
assert "gather_and_maybe_dequant_cache" in dst, (
    "the quantised-KV gather is gone from the DCP prefill loop; with "
    "kv-cache-dtype fp8 this arm has no gather to measure"
)
print("=== v4-rev51739 verified: cp_gather_cache is the pre-#51739 form ===")
PY

bash "$(dirname "${BASH_SOURCE[0]}")/k3_gather_microbench.sh"
