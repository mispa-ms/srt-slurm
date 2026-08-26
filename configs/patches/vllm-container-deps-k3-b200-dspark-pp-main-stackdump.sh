#!/usr/bin/env bash
# The DSpark-under-PP chain, plus a periodic all-thread stack dump.
# =============================================================================
# WHAT THIS IS FOR. On the current nightly the ns7 PP2 arms deadlock at startup.
# Both stages stay in lockstep through kernel_warmup.py:194 and then split on the
# next line, capture_model():
#
#   stage 0   17:48:15 JIT warmup -> 17:48:36 Graph capturing finished    (ok)
#   stage 1   17:48:15 JIT warmup -> silence                              (hung)
#
# Stage 0 goes on and dies 30 minutes later inside a gloo *send*, so it is waiting
# in a collective that stage 1 never reaches. Stage 1 is therefore stuck on
# something local. It is the only stage that builds a drafter -- our PP change puts
# the speculator on the last stage only, and the [emptycache] line confirms it fires
# there and nowhere else -- so the suspect is:
#
#   if self.speculator is not None:
#       with use_workspace_lane(self._draft_workspace_lane):
#           self.speculator.capture()
#
# It is not memory: empty_cache leaves 75.43 of 178.35 GiB free. It is not a caught
# error: stage 1 logs nothing at all, no warning, no symm-mem chatter. The logs are
# out of answers, hence a stack.
#
# WHY NOT VLLM_TRACE_FUNCTION. vLLM's own hang-debugging knob writes per-process
# trace files under tempfile.gettempdir() inside the container
# (vllm/config/vllm.py:801). Nothing collects that path, so the output dies with the
# job. faulthandler writes to stderr, which srtslurm already captures into
# prenyx<node>_agg_w0.out.log and the AIB archiver already pulls down.
#
# HOW. sitecustomize is imported by every Python process that starts with site
# enabled, so one file covers the API server, the engine core and all sixteen
# workers without touching vLLM. dump_traceback_later(repeat=True) then prints every
# thread of every rank on a timer. A healthy rank prints its stack too -- that is the
# point, the two stages are compared against each other.
#
# K3_STACKDUMP_SECONDS gates it, so this script is inert unless a config asks. Set it
# to something shorter than the 1800 s gloo timeout and longer than a normal capture,
# and expect several dumps: the hang opens about 13 minutes in and lasts until the
# timeout fires.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dspark-pp-main.sh

echo "=== stackdump: install the periodic all-thread dump ==="

python3 - <<'PY'
import os
import site
import sys

# site-packages, the same directory vllm itself lives in, is on sys.path for every
# interpreter here, so sitecustomize placed there is picked up by all of them.
candidates = [p for p in (site.getsitepackages() or []) if p.endswith("dist-packages")]
if not candidates:
    candidates = site.getsitepackages() or []
if not candidates:
    sys.exit("[stackdump] FATAL: no site-packages directory to install into")
target = os.path.join(candidates[0], "sitecustomize.py")

BLOCK = '''
# --- k3 stackdump ---------------------------------------------------------
# Dump every thread of this process on a timer when K3_STACKDUMP_SECONDS is set.
# Output goes to stderr so it lands in the worker log the harness already keeps.
def _k3_install_stackdump():
    import os

    seconds = os.environ.get("K3_STACKDUMP_SECONDS")
    if not seconds:
        return
    try:
        seconds = float(seconds)
    except ValueError:
        return
    if seconds <= 0:
        return
    import faulthandler
    import sys

    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(
        seconds, repeat=True, file=sys.stderr, exit=False
    )


_k3_install_stackdump()
# --- end k3 stackdump -----------------------------------------------------
'''

existing = ""
if os.path.exists(target):
    existing = open(target).read()
    if "k3 stackdump" in existing:
        print("[stackdump] already installed: " + target)
        raise SystemExit(0)
    # Preserve whatever the image shipped; append rather than replace.
    print("[stackdump] appending to an existing sitecustomize: " + target)

src = existing + BLOCK
compile(src, target, "exec")
open(target, "w").write(src)
print("[stackdump] installed: " + target)
PY

# Prove the hook actually runs in a fresh interpreter rather than trusting the write.
# A one-second timer with a short sleep makes the dump appear in this setup log too,
# which is the cheapest possible end-to-end check.
if K3_STACKDUMP_SECONDS=1 python3 -c "import time; time.sleep(2)" 2>&1 | grep -q "Timeout (0:00:01)"; then
    echo "[stackdump] verified: a fresh interpreter dumps on the timer"
else
    echo "[stackdump] FATAL: sitecustomize did not fire in a fresh interpreter" >&2
    exit 1
fi

echo "=== stackdump: done ==="
