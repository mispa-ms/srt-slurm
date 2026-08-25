#!/usr/bin/env bash
# Kimi-K3-NVFP4 on GB300: JET checkpoint + vLLM PR #53324.
# =============================================================================
# Two things before the server starts.
#
#   1. The HF cache shim, so nvidia/Kimi-K3-NVFP4 resolves to the JET copy rather
#      than a 1.6 TB download inside the job.
#
#   2. vLLM PR #53324, "[KV Connector] Support MooncakeStore with hybrid DCP prefix
#      caching". This is the patch Wei Zhao is running on top of vLLM main for
#      nvfp4 + trtllm-moe + dep16 (#kimi-k3 thread, 2026-08-24). It fixes the load
#      path deriving a partially-reused block's key from hit_length, which under
#      fine-grained prefix hits requests a key the block was not stored under and
#      makes Mooncake return -7.
#
# WHY A PATCH AND NOT A SOURCE BUILD. All 13 files the PR touches are pure Python;
#   it adds no kernels and changes nothing compiled. Applying the diff to the
#   installed package therefore produces the same runtime as building an image from
#   a branch, minus one to two hours of aarch64 build we have never exercised (the
#   source-build path in AGG_V2_REPRO is x86 only).
#
# HOW. patch-ng, a pure-Python applier, so the job does not depend on git or patch
#   being present in the runtime image -- the K3 images have neither. git apply is
#   the fallback for the case where pip cannot reach the index. tests/ hunks are
#   dropped: the image ships no test tree, and a failed hunk there would abort an
#   otherwise correct patch.
#
# The PR is OPEN, not merged. It is pinned by head sha so that a force-push to the
#   author's branch cannot silently change what this arm measured. Moving to a newer
#   head is a deliberate edit here, not a surprise at run time.
#
# Applying is verified, not assumed: the marker below must be absent before and
#   present after, or the job dies here rather than reporting numbers from an
#   unpatched engine.
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps-k3nvfp4-hfshim.sh ]]; then
    bash /configs/patches/vllm-container-deps-k3nvfp4-hfshim.sh
fi

PR_NUM=53324
PR_HEAD=574d2e4092d96bf2bac99e9dfa1ffecf39de334e
# Added by the PR in mooncake/store/protocol.py; absent from main.
MARKER=TAIL_KEY_BOUNDARY_ENTRY_SIZE

# Dump what the container can actually see of the fabric and the host. Mooncake
# names its RDMA devices by hand (device_name in the recipe) and reports a wrong
# name only as "Found 0 HCAs" / "No available RNIC", which does not say what the
# right name would have been. Cheap, once per node, and oci-aga's NIC inventory is
# unrecorded in our tree.
echo "=== k3nvfp4-pr$PR_NUM: fabric and host inventory ==="
echo "--- ibv_devices"; ibv_devices 2>&1 | head -20 || echo "(ibv_devices unavailable)"
echo "--- /sys/class/infiniband"; ls -1 /sys/class/infiniband 2>&1 | head -20 || echo "(none)"
echo "--- MemTotal"; grep -E "MemTotal|MemAvailable" /proc/meminfo || true
echo "--- vllm"; python3 -c "import vllm; print(vllm.__version__, vllm.__file__)" || true
echo "=== k3nvfp4-pr$PR_NUM: inventory done ==="

SITE=$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
PROTO="$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/protocol.py"

if grep -q "$MARKER" "$PROTO" 2>/dev/null; then
    echo "[k3nvfp4-pr$PR_NUM] already applied (marker present in protocol.py)"
    exit 0
fi

echo "=== k3nvfp4-pr$PR_NUM: applying PR #$PR_NUM @ $PR_HEAD ==="

DIFF=/tmp/pr${PR_NUM}.diff
RAW=/tmp/pr${PR_NUM}.raw.diff
curl -fsSL --retry 5 --retry-delay 5 \
    "https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/${PR_NUM}.diff" -o "$RAW"

# Keep only the vllm/ file sections. The image has no tests/ tree.
python3 - "$RAW" "$DIFF" <<'PY'
import sys
raw, out = sys.argv[1], sys.argv[2]
keep, cur = [], False
for line in open(raw):
    if line.startswith("diff --git a/"):
        cur = line.split(" b/", 1)[-1].startswith("vllm/")
    if cur:
        keep.append(line)
open(out, "w").writelines(keep)
n = sum(1 for line in keep if line.startswith("diff --git a/"))
print(f"[filter] kept {n} vllm/ file sections")
if n != 8:
    sys.exit(f"[filter] FATAL: expected 8 vllm/ files, got {n} — the PR changed shape")
PY

# git apply first: it is the applier the diff was checked against before submission,
# and the only one that reports which hunk failed. The image ships neither git nor
# patch, so install whichever is missing -- the benchmark stage already apt-gets git
# in these containers, so the index is reachable.
if ! command -v git >/dev/null 2>&1 || ! command -v patch >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq git patch >/dev/null 2>&1 || true
fi

applied=0

# Run from $SITE, not `git -C $SITE`: -C makes git treat it as a repository and it is
# not one, which fails before the patch is even read. Plain `git apply` in a non-repo
# directory patches files in place, which is what we want.
if [[ "$applied" -eq 0 ]] && command -v git >/dev/null 2>&1; then
    echo "[k3nvfp4-pr$PR_NUM] trying git apply"
    if (cd "$SITE" && git apply --check -p1 -v "$DIFF") && (cd "$SITE" && git apply -p1 "$DIFF"); then
        applied=1
        echo "[k3nvfp4-pr$PR_NUM] applied with git apply"
    fi
fi

if [[ "$applied" -eq 0 ]] && command -v patch >/dev/null 2>&1; then
    echo "[k3nvfp4-pr$PR_NUM] trying patch -p1"
    if (cd "$SITE" && patch -p1 --forward --batch < "$DIFF"); then
        applied=1
        echo "[k3nvfp4-pr$PR_NUM] applied with patch"
    fi
fi

if [[ "$applied" -eq 0 ]] && python3 -m pip install --quiet --disable-pip-version-check patch-ng 2>/dev/null; then
    echo "[k3nvfp4-pr$PR_NUM] trying patch-ng"
    if python3 - "$DIFF" "$SITE" <<'PY'
import logging, sys, patch_ng
# patch-ng reports every failure through this logger and configures no handler of
# its own, so without this an unapplied hunk is a bare False and nothing else.
logging.basicConfig(level=logging.DEBUG, format="[patch-ng] %(levelname)s %(message)s")
ps = patch_ng.fromfile(sys.argv[1])
if not ps:
    sys.exit("[patch-ng] could not parse the diff")
sys.exit(0 if ps.apply(strip=1, root=sys.argv[2]) else "[patch-ng] apply returned False")
PY
    then
        applied=1
        echo "[k3nvfp4-pr$PR_NUM] applied with patch-ng"
    fi
fi

if [[ "$applied" -eq 0 ]]; then
    echo "[k3nvfp4-pr$PR_NUM] FATAL: no applier succeeded (git apply, patch, patch-ng)." >&2
    echo "[k3nvfp4-pr$PR_NUM] Installed vLLM: $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1)" >&2
    echo "[k3nvfp4-pr$PR_NUM] Expected the tree of the commit the image tag names." >&2
    exit 1
fi

# The applier reporting success is not the same as the change being there.
if ! grep -q "$MARKER" "$PROTO"; then
    echo "[k3nvfp4-pr$PR_NUM] FATAL: applier claimed success but $MARKER is absent." >&2
    exit 1
fi
python3 -c "import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol" \
    || { echo "[k3nvfp4-pr$PR_NUM] FATAL: patched module does not import." >&2; exit 1; }

echo "=== k3nvfp4-pr$PR_NUM: applied and verified ==="
