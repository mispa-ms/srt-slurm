#!/usr/bin/env bash
# Wei's GB300 stack, unmodified, so a B300 run reproduces his numbers.
# =============================================================================
# WHAT THIS IS. NVIDIA/InferenceMAX PR #265 (branch wzhao/k3-gb300-test-2) is the
# GB300 Kimi-K3 result that is currently the frontier: its disagg 1P1D points at
# c48-c56 sit around 11,000 tok/s/chip, above every aggregated point on the same
# chart. Po-Han reproduced it on AWS-CMH from the same recipes. The open question,
# asked in #kimi-k3 by Itay, is whether B300 disagg looks like that curve.
#
# The only way that question gets a clean answer is to change as little as
# possible. So this carries Wei's artefacts verbatim:
#
#   container   vllm/vllm-openai:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
#   patch       vllm-k3-nvfp4-perf-3696c772-on-46638857.patch, byte for byte,
#               copied out of his PR as k3-wei-infmax-3696c772-on-46638857.patch
#   dynamo      ba83080ecd31c1ce918559e576d3c5bc9e092ff1
#
# 6,368 lines over 73 files. Verified to apply clean with `git apply --check`
# against a worktree at 46638857fd, which is the vLLM commit his container is
# built from -- the "on-46638857" in the patch name is a claim, and this is the
# check of it.
#
# NOT OUR STACK. None of our own carries are applied here: not the v8 patch, not
# #53324, not the compact-I/O port, not the int64idx / mambacache / dspark-pp
# scripts. Wei's branch already contains its own answers to those, and mixing the
# two would produce a third thing that reproduces neither. The B300 arms that use
# our stack keep using kimi-k3-nightly-v8*.sh and are unaffected by this file.
#
# ON THE NAME. His branch is called nvfp4-perf, and the recipes are labelled
# `precision: fp4`, but the checkpoint every one of them loads is
# `moonshotai/Kimi-K3`, which is MXFP4 -- the same checkpoint our B300 ladder
# runs. The patch contains no reference to nvidia/Kimi-K3-NVFP4 and no cast
# (`cast_mxfp4`, `nvfp4_quantize`: zero hits in 6,368 lines). So this is a
# platform port and nothing else, and its numbers compare directly to ours.
# =============================================================================
set -euo pipefail

PATCH=/configs/patches/k3-wei-infmax-3696c772-on-46638857.patch
SITE=$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin)))")

echo "=== wei-gb300: applying $(basename "$PATCH") ==="

# Refuse on the wrong image rather than fuzzing 73 files into it. vLLM's dev
# version carries the commit as "+g<sha>".
python3 - <<'PY'
import re
import sys

import vllm

want = "46638857"
v = getattr(vllm, "__version__", "")
m = re.search(r"\+g([0-9a-f]{7,})", v)
got = m.group(1) if m else ""
if not got.startswith(want[: len(got)]) and not want.startswith(got):
    sys.exit(f"[wei-gb300] FATAL: this patch is generated against {want}; the "
             f"image reports {v!r}. Applying it elsewhere would fuzz 73 files.")
print(f"[wei-gb300] image {v} matches the patch base {want}")
PY

cd "$SITE"
if [ -f ".wei-gb300-applied" ]; then
  echo "[wei-gb300] already applied"
else
  git apply -p1 --whitespace=nowarn "$PATCH" 2>/dev/null \
    || patch -p1 --forward --no-backup-if-mismatch --fuzz=0 < "$PATCH"
  date -u +%Y-%m-%dT%H:%M:%SZ > ".wei-gb300-applied"
  echo "[wei-gb300] applied"
fi

python3 - <<'PY'
import importlib.util
import os
import subprocess
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))

# The patch is 73 files; a per-file check would be a paraphrase of it. Import the
# modules a K3 disagg worker actually loads instead -- that is what caught the
# missing KVCacheGroupSpec import when we ported his Mooncake commits by hand.
for mod in (
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker",
    "vllm.v1.attention.backends.mla.tokenspeed_mla",
    "vllm.v1.worker.gpu.spec_decode.dspark.utils",
    "vllm.models.kimi_k3.nvidia.kda",
):
    __import__(mod)

# And the compact-I/O flags his recipes turn on must be readable, or the config
# sets something inert and the run is not his run.
worker = open(os.path.join(
    root, "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py")).read()
for needle in ('extra_config.get("compact_group_io"',
               'extra_config.get("max_load_batch_keys")'):
    if needle not in worker:
        sys.exit(f"[wei-gb300] FATAL: {needle} missing; his recipes set that flag")

print("[wei-gb300] verified: worker modules import, both compact-I/O flags present")
PY

echo "=== wei-gb300: done ==="
