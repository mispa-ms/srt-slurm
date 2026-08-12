#!/usr/bin/env bash
# k3-merged-v4 + Mooncake under DCP, with the gather microbenchmark attached.
#
# Identical to kimi-k3-merged-v4-mooncake.sh in everything that touches the
# server. The only addition is the kernel timing run, which happens before the
# model loads and exists so the #51739 A/B has a direct measurement and not only
# an end-to-end one.
#
# The twin on the other side is kimi-k3-merged-v4-rev51739-mooncake.sh, which
# runs the same benchmark against the pre-#51739 kernel.
#
# WHY THIS ASSERTS THE COMMIT IS PRESENT. The image is chosen by a pipeline-level
# variable, one per pipeline, so this arm and its reverted twin cannot be
# submitted together. Nothing else here would notice if they were: the v4 marker
# that kimi-k3-merged-v4.sh checks is present on both images, because reverting
# #51739 does not touch it. Without the check below, running these arms on the
# reverted image would compare that image against itself and produce a clean,
# meaningless null result.
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
assert dst["cp_gather_cache"] == "workspace[:toks]", (
    f"wrong image: this arm is the #51739-present side of the A/B, but "
    f"cp_gather_cache passes dst={dst['cp_gather_cache']}, which is the "
    f"pre-#51739 form. This is the reverted image, and running here would "
    f"compare it against itself."
)
print("=== v4 verified: cp_gather_cache carries the #51739 slice ===")
PY

bash "$(dirname "${BASH_SOURCE[0]}")/k3_gather_microbench.sh"
