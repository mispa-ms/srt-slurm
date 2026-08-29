#!/usr/bin/env bash
# vLLM PR #53598 -- serve K3 local prefix-cache hits correctly under DCP.
# =============================================================================
# WHY THIS ARM EXISTS. On this workstream the AgentX agentic trace is 99.3% input
# tokens (18,687 vs 131.9 tok/s/gpu on the c48 ns=4 arm), so
#
#     y = P / (1 - h) / N_gpu
#
# is not an approximation, it is very nearly an identity. The measured true miss on
# our best arm is 6.68% -- 84.90% served from HBM, 15.10% falling through, of which
# Mooncake serves 55.73%. Anything that raises h is multiplicative on y; a kernel that
# raises P is additive. That is the whole reason this and its two sibling patches were
# picked over the kernel PRs that landed the same week.
#
# WHAT IT CHANGES. Upstream #53598 is titled for ROCm but touches no kernel and no
# platform code -- only vllm/v1/core/kv_cache_coordinator.py and
# vllm/v1/core/kv_cache_utils.py, +37/-10. It defines prefix-cache lookup, hit
# reconciliation and block ownership for hybrid KV-cache layouts under decode context
# parallelism:
#
#   - hybrid DCP prefix-cache geometry
#   - fine-grained Mamba/full-attention cache alignment
#   - stale partial-hit CoW source refresh in the shared KV-cache manager
#
# We run decode-context-parallel-size 8 with a hybrid Mamba+MLA layout, so this is our
# exact configuration.
#
# ORDERING. It must run after k3-mooncake-53324-828.patch, which edits both of the same
# two files. Dry-run on a bare 6f7df92a tree and again on 6f7df92a+53324: clean in both,
# 0 rejects, one hunk at fuzz 1. Chain scripts place it last for that reason.
#
# WHAT IT IS NOT. Not merged upstream -- #53598 is open. It is measured here as one
# patch on one arm, against the same config without it, because a stack of unmerged
# patches priced together tells you nothing about which one paid.
# =============================================================================
set -euo pipefail

echo "=== k3-pr53598: serve prefix cache hits under DCP ==="

VLLM_ROOT=${VLLM_ROOT:-$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')}
command -v patch >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq patch; }

cd "$VLLM_ROOT"

if grep -q "dcp_world_size_for_kv_cache_spec" vllm/v1/core/kv_cache_utils.py 2>/dev/null; then
  echo "[pr53598] already applied"
else
  # Fail loudly rather than half-apply: a partially patched cache coordinator would
  # serve wrong blocks under a valid prefix hash, which is a silent-accuracy failure,
  # not a crash.
  patch -p1 --forward --dry-run < /configs/patches/k3-pr53598-828.patch
  patch -p1 --forward < /configs/patches/k3-pr53598-828.patch
  echo "[pr53598] applied"
fi

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
for rel in ("vllm/v1/core/kv_cache_coordinator.py", "vllm/v1/core/kv_cache_utils.py"):
    src = open(os.path.join(root, rel)).read()
    compile(src, rel, "exec")

# The substance of the patch is one helper: full-attention/MLA groups are sharded
# across DCP ranks, but Mamba, sliding-window and chunked-local state is replicated
# per rank. Before this, every group took the process DCP size, so a replicated group's
# block geometry and prefix hashing were computed as if it were sharded 8 ways -- which
# is a prefix-cache *hit* bug on a hybrid layout, not a crash. Assert the helper is
# there and that the coordinator actually routes through it.
import vllm.v1.core.kv_cache_utils as ku

if not hasattr(ku, "dcp_world_size_for_kv_cache_spec"):
    sys.exit("[pr53598] FATAL: dcp_world_size_for_kv_cache_spec missing after patch")

coord = open(os.path.join(root, "vllm/v1/core/kv_cache_coordinator.py")).read()
if "dcp_world_size_for_kv_cache_spec" not in coord:
    sys.exit("[pr53598] FATAL: the coordinator does not call the new helper")

import vllm.v1.core.kv_cache_coordinator  # noqa: F401

print("[pr53598] verified: helper present, coordinator routes through it, both import")
PY

echo "=== k3-pr53598: done ==="
