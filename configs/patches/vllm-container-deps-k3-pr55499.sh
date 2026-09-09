#!/usr/bin/env bash
# Cherry-pick vllm#55499 at runtime: the TRTLLM ragged prefill regression fix.
# =============================================================================
# WHAT. FlashInfer 0.6.18 (vLLM nightlies since 2026-08-30, #54313) made
# trtllm_ragged_attention_deepseek evaluate an "inactive rows" check on the device
# on every call -- bool((~active_rows).any().item()) -- one D2H sync per MLA layer per
# prefill step. 0.6.17 had no such check. FlashInfer 0.6.18.post1 (deae5b8b) adds
# skip_all_rows_active_check; vllm#55499 (Wei Zhao, merged 2026-09-09 20:38 UTC as
# 8c87c333b8) validates once on the host from query_lens_cpu and passes the flag.
#
# HOW. Two pure-Python edits, both gated by a --dry-run --fuzz=0 patch:
#   1. FlashInfer: the prefill.py hunk of post1 (deae5b8b), applied to the installed
#      0.6.18 package in place. Only when the installed version is exactly 0.6.18;
#      a post1-or-newer image already has it and is left alone.
#   2. vLLM: the single commit 8c87c333b8 restricted to mla_attention.py and
#      prefill/trtllm_ragged.py (git apply --check passes on nightly 385dce36).
# K3_VLLM_ROOT / K3_FI_ROOT override the discovered package roots (local testing).
# =============================================================================
set -euo pipefail

echo "=== pr55499: TRTLLM ragged prefill regression fix (FlashInfer post1 hunk + vllm#55499) ==="

VLLM_ROOT=${K3_VLLM_ROOT:-$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')}
FI_ROOT=${K3_FI_ROOT:-$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("flashinfer").origin)))')}
FI_PREFILL="$FI_ROOT/flashinfer/prefill.py"
TR="$VLLM_ROOT/vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py"
[ -f "$FI_PREFILL" ] || { echo "[pr55499] FATAL: no flashinfer/prefill.py under $FI_ROOT" >&2; exit 1; }
[ -f "$TR" ] || { echo "[pr55499] FATAL: no trtllm_ragged.py under $VLLM_ROOT" >&2; exit 1; }

# FlashInfer version without importing it (import initialises CUDA libraries).
FI_VER=$(python3 - "$FI_ROOT" <<'PY'
import os, re, sys
root = sys.argv[1]
import glob
for d in glob.glob(os.path.join(root, "flashinfer_python-*.dist-info")) + glob.glob(os.path.join(root, "flashinfer-*.dist-info")):
    for line in open(os.path.join(d, "METADATA")):
        if line.startswith("Version:"):
            print(line.split(":", 1)[1].strip()); sys.exit(0)
for rel in ("flashinfer/version.py", "flashinfer/_version.py", "version.txt", "flashinfer/version.txt"):
    p = os.path.join(root, rel)
    if os.path.exists(p):
        s = open(p).read()
        m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", s) or re.search(r"^\s*([0-9][0-9A-Za-z.+-]*)\s*$", s, re.M)
        if m:
            print(m.group(1)); sys.exit(0)
import glob
for d in glob.glob(os.path.join(root, "flashinfer_python-*.dist-info")) + glob.glob(os.path.join(root, "flashinfer-*.dist-info")):
    for line in open(os.path.join(d, "METADATA")):
        if line.startswith("Version:"):
            print(line.split(":", 1)[1].strip()); sys.exit(0)
print("unknown")
PY
)
echo "[pr55499] flashinfer $FI_VER under $FI_ROOT"

# 1. FlashInfer prefill.py. The nightly builds FlashInfer from source and its version.py
# reads "0.0.0+unknown", so the version string is informational only; the gate is the
# exact-context dry-run of the post1 hunk (3 hunks, fuzz 0) -- it applies to 0.6.18 and
# to nothing else, which is a stricter check than any version string.
if grep -q "skip_all_rows_active_check" "$FI_PREFILL"; then
    echo "[pr55499] flashinfer already has skip_all_rows_active_check (>= 0.6.18.post1); skipping the FlashInfer hunk"
else
    if ! patch -p1 -d "$FI_ROOT" --dry-run --forward --fuzz=0 < /configs/patches/flashinfer-0.6.18.post1-prefill.patch > /tmp/pr55499-fi-dry.log 2>&1; then
        echo "[pr55499] FATAL: the post1 prefill.py hunk does not apply to this flashinfer ($FI_VER); it is cut against 0.6.18" >&2; cat /tmp/pr55499-fi-dry.log >&2; exit 1
    fi
    patch -p1 -d "$FI_ROOT" --forward --fuzz=0 < /configs/patches/flashinfer-0.6.18.post1-prefill.patch
    echo "[pr55499] applied the FlashInfer 0.6.18.post1 prefill.py hunk in place (flashinfer reported $FI_VER)"
fi

# 2. vLLM
if grep -q "skip_all_rows_active_check=True" "$TR"; then
    echo "[pr55499] vllm already carries #55499; skipping"
else
    if ! patch -p1 -d "$VLLM_ROOT" --dry-run --forward --fuzz=0 < /configs/patches/k3-pr55499.patch > /tmp/pr55499-vllm-dry.log 2>&1; then
        echo "[pr55499] FATAL: #55499 does not apply to this vllm (cut against 385dce36)" >&2; cat /tmp/pr55499-vllm-dry.log >&2; exit 1
    fi
    patch -p1 -d "$VLLM_ROOT" --forward --fuzz=0 < /configs/patches/k3-pr55499.patch
    echo "[pr55499] applied vllm#55499 under $VLLM_ROOT"
fi

python3 - "$FI_PREFILL" "$TR" "$VLLM_ROOT/vllm/model_executor/layers/attention/mla_attention.py" <<'PY'
import sys
fi, tr, mla = (open(p).read() for p in sys.argv[1:4])
for p in sys.argv[1:4]:
    compile(open(p).read(), p, "exec")
problems = []
if "skip_all_rows_active_check" not in fi: problems.append("flashinfer prefill.py lacks skip_all_rows_active_check")
if tr.count("skip_all_rows_active_check=True") < 2: problems.append("vllm trtllm_ragged.py does not pass the skip flag on both call sites")
if "query_lens_cpu" not in tr or "query_lens_cpu=prefill_query_lens_cpu" not in mla: problems.append("query_lens_cpu plumbing missing")
if "all_rows_active" not in mla: problems.append("all_rows_active metadata missing")
if problems: sys.exit("[pr55499] FATAL: " + "; ".join(problems))
print("[pr55499] verified: FlashInfer skip flag present, vllm passes it on both prefill call sites, CPU query lengths plumbed; files compile")
PY
echo "=== pr55499: done ==="
