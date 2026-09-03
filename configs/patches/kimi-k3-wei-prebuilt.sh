#!/usr/bin/env bash
# Wei's stack from HIS prebuilt image, so nothing is applied at runtime.
# =============================================================================
# WHY THIS EXISTS AT ALL. Wei published the container his InferenceMAX run used:
#
#   vllm/vllm-openai:nightly-dev-x86_64-cu13-3696c77
#   vllm/vllm-openai:nightly-dev-arm64-cu13-3696c77
#
# `3696c77` is the head of his branch, which is the same commit our patch is
# named for -- k3-wei-infmax-3696c772-on-46638857.patch is that commit's diff
# against base 46638857. So the image already contains, byte for byte, what
# kimi-k3-wei-gb300.sh spends six thousand lines applying. Running that script
# here would try to apply a patch to a tree that already has it and fail.
#
# But "the tag says 3696c77" is a claim about a string, not about the tree, and
# a silently-unpatched image would produce a full hour of plausible numbers that
# reproduce nothing. So this script applies NOTHING and checks EVERYTHING: three
# symbols the patch introduces and cannot be in the base, plus the five modules a
# K3 worker imports. It is a gate, not a build step.
#
# The three markers are classes the patch adds, chosen from different files so a
# partial image cannot pass:
#
#   TailKeyBoundary            the eagle/tail-boundary work
#   KimiFusedSharedExpert      the K3 model change
#   MooncakeLookupResult       the connector I/O change
# =============================================================================
set -euo pipefail

SITE=$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin)))")

echo "=== wei-prebuilt: verifying the image already carries his branch ==="
python3 -c "import vllm; print('    vllm', vllm.__version__)" || true

python3 - "$SITE" <<'PY'
import pathlib, sys, subprocess

site = pathlib.Path(sys.argv[1]) / "vllm"
# Grep the installed tree rather than importing: a missing symbol should read as
# "this image is not his", not as an ImportError from some unrelated dependency.
markers = ["TailKeyBoundary", "KimiFusedSharedExpert", "MooncakeLookupResult"]
missing = []
for m in markers:
    r = subprocess.run(["grep", "-rlq", m, str(site)])
    print(f"    {m:<26} {'found' if r.returncode == 0 else 'MISSING'}")
    if r.returncode:
        missing.append(m)
if missing:
    sys.exit(
        "wei-prebuilt: this image does not carry his branch -- missing "
        + ", ".join(missing)
        + ".\nExpected vllm/vllm-openai:nightly-dev-<arch>-cu13-3696c77 or an "
          "image built from the same commit. Refusing rather than running an "
          "hour that reproduces nothing.")
PY

# The same five a K3 disagg/agg worker loads. Import errors here are the ones
# that otherwise surface an hour later as a dead server.
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
    sys.exit("wei-prebuilt: import check failed\n  " + "\n  ".join(bad))
PY

echo "=== wei-prebuilt: done, nothing applied ==="
