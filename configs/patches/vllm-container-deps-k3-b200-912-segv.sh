#!/usr/bin/env bash
# Kimi-K3 B200 AGG, 09/12-or-newer nightly: the 912 chain plus fault-capture instrumentation.
# =============================================================================
# 67590200 crashed 4/4 of the SimpleCPUOffload + DSpark arms the same way: PP1 segfaults on all
# eight ranks 32-54 min into serving, on four different nodes, while the Mooncake twins and the
# c96 no-spec SimpleCPU arm served clean. vLLM's own handler printed
#     !!!!!!! Segfault encountered !!!!!!!
#       File "<unknown>", line 0, in 0xffffffffffffffff
# i.e. the faulting thread had no Python frame, so we learned nothing about where it died.
#
# This chain adds two things and changes no vLLM behaviour:
#   1. sitecustomize.py -- imported automatically by every Python process in the container,
#      including each VllmWorker. It calls faulthandler.enable(all_threads=True) so SIGSEGV
#      dumps EVERY thread's Python stack, not just the faulting one. Even when the faulting
#      thread is native, the other threads name the subsystem that was active (the offload
#      worker, the Mooncake store thread, the drafter).
#   2. A best-effort core dump: `ulimit -c unlimited` and a core_pattern pointed at the job's
#      log directory. core_pattern is a host sysctl, so this only takes if the container is
#      privileged; the script reports which of the two it got and never fails on it.
#
# Everything else is `vllm-container-deps-k3-b200-912.sh` verbatim -- this runs it first, so the
# upstream-presence FATALs and the draft-loader guard still apply unchanged and still abort here.
set -euo pipefail

echo "=== k3-b200-912-segv: running the 912 chain first ==="
bash /configs/patches/vllm-container-deps-k3-b200-912.sh

echo "=== k3-b200-912-segv: fault capture ==="

SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
# Refuse to silently replace an existing sitecustomize: chaining to it is the only safe move,
# and if one appears later we want to know rather than to have swallowed it.
if [ -e "${SITE}/sitecustomize.py" ]; then
    if grep -q "K3_SEGV_DIR" "${SITE}/sitecustomize.py"; then
        echo "[segv] our sitecustomize is already installed; leaving it"
    else
        echo "[segv] FATAL: ${SITE}/sitecustomize.py already exists and is not ours -- chain to it instead of overwriting" >&2
        exit 1
    fi
fi
cat > "${SITE}/sitecustomize.py" <<'PY'
"""Dump every thread's Python stack on a fatal signal.

vLLM enables faulthandler for the current thread only, which on the 67590200 crash printed a
single `<unknown>` frame. all_threads=True costs nothing at runtime and names whatever the
other threads were doing when the native fault hit.
"""
import faulthandler
import os
import sys

try:
    faulthandler.enable(file=sys.stderr, all_threads=True)
    # Keep a copy per pid: stderr interleaves badly across 8 workers on one node.
    _d = os.environ.get("K3_SEGV_DIR")
    if _d:
        os.makedirs(_d, exist_ok=True)
        _f = open(os.path.join(_d, "faulthandler-%d.txt" % os.getpid()), "w")
        faulthandler.enable(file=_f, all_threads=True)
except Exception as exc:  # never break the interpreter over diagnostics
    print("[segv] faulthandler setup failed: %r" % (exc,), file=sys.stderr)
PY
python3 -c "import sitecustomize, faulthandler, sys; assert faulthandler.is_enabled(); print('[segv] sitecustomize active at', sitecustomize.__file__)"

# Best effort only: both of these need privileges the container may not have.
if ulimit -c unlimited 2>/dev/null; then
    echo "[segv] core size limit: $(ulimit -c)"
else
    echo "[segv] could not raise the core size limit (not privileged); relying on faulthandler"
fi
CORE_DIR="${K3_SEGV_DIR:-/logs}"
if echo "${CORE_DIR}/core.%e.%p" > /proc/sys/kernel/core_pattern 2>/dev/null; then
    echo "[segv] core_pattern -> $(cat /proc/sys/kernel/core_pattern)"
else
    echo "[segv] core_pattern is read-only here (host sysctl); faulthandler is the fallback"
fi

echo "=== k3-b200-912-segv: done ==="
