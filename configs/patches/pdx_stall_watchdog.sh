#!/bin/bash
# Dump every worker's stack the moment the engine stalls, and get out of the way.
#
# WHY. On aws-pdx the Kimi-K3 c70 run freezes shortly after the profiling phase
# begins -- three times now, at three different loads, with and without
# VLLM_USE_BREAKABLE_CUDAGRAPH and compact_group_io. The engine blocks in
# shm_broadcast.dequeue waiting for a worker reply and only reports it five
# minutes later as "RPC call to sample_tokens timed out", by which point the
# workers have been torn down and there is nothing left to look at. The step it
# dies on is pure decode and tiny (18 requests, one token each) but carries very
# long contexts -- one request at 309,900 tokens.
#
# So: catch it live. vLLM prints the marker below from the engine process about
# 60 s into the stall, four minutes before it gives up. That is the window.
#
# This is diagnostics only. It must never fail a run: every step is best-effort
# and the script always exits 0.
set -u
LOGDIR=${1:-/logs}
# Two triggers. The second is the better one -- run 417809 showed the wedge
# starts as a Mooncake transfer that never completes, and the engine only
# notices minutes later -- but keep the engine-side marker too in case a stall
# ever arrives without it.
MARKER="No available shared memory broadcast block found"
MARKER2="Failed to complete transfers after"
HOST=$(hostname)

# One watchdog per PID NAMESPACE, not per node.
#
# The first version claimed a single lock in $LOGDIR -- which both containers on
# the node share -- so whichever started first won. In run 418400 that was the
# frontend container, and from there the vLLM workers are in another namespace:
# pgrep matched one pid, /proc/<pid>/exe came back "Permission denied", and
# py-spy got nothing. (nvidia-smi still saw all eight worker pids, because it
# asks the driver rather than /proc, which is how that run produced GPU
# utilisation and no stacks.)
#
# Let every container arm its own. Duplicate dumps cost a few seconds; a dump
# taken from the wrong namespace is worthless.
NS=$(readlink /proc/self/ns/pid 2>/dev/null | tr -dc '0-9')
NS=${NS:-$$}
TAG=$HOST-ns$NS
STAMP=$LOGDIR/.pyspy-watchdog-$TAG.claimed
if ! (set -o noclobber; : > "$STAMP") 2>/dev/null; then
    exit 0
fi

exec >> "$LOGDIR/pyspy-watchdog-$TAG.log" 2>&1
echo "[watchdog] start $(date -u +%FT%TZ) on $HOST, watching $LOGDIR"

if ! command -v py-spy > /dev/null 2>&1; then
    python3 -m pip install --no-deps -q py-spy || {
        echo "[watchdog] py-spy will not install; nothing to do"; exit 0; }
fi

# py-spy needs ptrace, so check it now rather than during the stall.
#
# Test against a real PYTHON process. The first version of this aimed py-spy at
# its own shell ($$) and read the inevitable failure as "no CAP_SYS_PTRACE" --
# py-spy only inspects Python interpreters, so that test fails on a healthy node
# too. On aws-pdx the permissions are in fact fine: kernel.yama.ptrace_scope is
# 0, which permits same-uid ptrace without CAP_SYS_PTRACE (CapEff is all zeros).
python3 -c 'import time; time.sleep(60)' &
probe=$!
sleep 2
if py-spy dump --pid "$probe" > /dev/null 2>&1; then
    echo "[watchdog] ptrace works (dumped a live python child)"
else
    echo "[watchdog] WARNING: py-spy cannot dump a plain python child:"
    py-spy dump --pid "$probe" 2>&1 | sed 's/^/[watchdog]   /' | head -5
    echo "[watchdog] Check kernel.yama.ptrace_scope and CAP_SYS_PTRACE."
    echo "[watchdog] Continuing anyway -- the /proc fallback still prints something."
fi
kill "$probe" 2>/dev/null || true

dump_everything() {
    # Two `local` statements, not one. Bash expands EVERY argument of `local`
    # before assigning any of them, so `local tag=$1 out="…$tag.txt"` reads $tag
    # while it is still unset -- which under `set -u` aborts the function and
    # takes the watchdog down with it. That is exactly how run 417809 produced
    # no dump at all.
    local tag=$1
    local out="$LOGDIR/pyspy-stall-$TAG-$tag.txt"
    {
        echo "=== $(date -u +%FT%TZ)  stall dump: $tag ==="
        echo
        # Unfiltered. A grep here once hid the very thing we were looking for:
        # run 418827 printed only supervisord/srun/EngineCore and it looked like
        # the workers were in another namespace, when in fact they were right
        # there under names the filter did not match.
        echo "--- processes (all) ---"
        ps -eo pid,ppid,stat,etime,pcpu,comm,args --sort=pid | head -60
        echo
        # vLLM titles its worker processes VLLM::Worker_TP0_DCP0, not
        # "VllmWorker" -- the pattern used in 418827 matched the EngineCore
        # alone and every worker stack was missed. VLLM:: catches both.
        local pids
        pids=$(pgrep -f 'VLLM::|VllmWorker|EngineCore' 2>/dev/null | tr '\n' ' ')
        echo "--- candidate pids: ${pids:-(none visible in this namespace)} ---"
        echo
        local got_stack=0
        for pid in $pids; do
            echo "--- py-spy dump --native --locals pid=$pid ---"
            if timeout 60 py-spy dump --pid "$pid" --native --locals 2>&1; then
                got_stack=1
            elif timeout 60 py-spy dump --pid "$pid" 2>&1; then
                got_stack=1
            else
                echo "  (py-spy failed for $pid)"
            fi
            echo
            echo "--- /proc/$pid/status wchan+state ---"
            grep -aE '^(State|Threads)' "/proc/$pid/status" 2>/dev/null
            echo
        done

        # If ptrace was refused -- wrong namespace, no CAP_SYS_PTRACE -- fall
        # back to faulthandler. With PYTHONFAULTHANDLER=1 set on the workers,
        # SIGABRT makes CPython print every thread's stack to its own stderr,
        # which lands in the worker log. It kills the process, so only do it on
        # the SECOND dump: by then the engine is already five minutes from its
        # own RPC timeout and the run is lost either way.
        if [[ "$got_stack" -eq 0 && "$tag" == "second" && -n "$pids" ]]; then
            echo "--- py-spy got nothing; SIGABRT for faulthandler stacks ---"
            echo "    (stacks appear in the worker log, not here)"
            for pid in $pids; do kill -ABRT "$pid" 2>/dev/null || true; done
        fi
        echo "--- nvidia-smi ---"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks_throttle_reasons.active \
                   --format=csv 2>&1
        echo
        echo "--- nvidia-smi compute processes ---"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
    } > "$out" 2>&1
    echo "[watchdog] wrote $out"
}

# Wait for the worker log to appear (model load is ~20 min, be patient).
W=""
for _ in $(seq 1 720); do
    W=$(ls "$LOGDIR"/*_agg_w0.out "$LOGDIR"/*_w0.out 2>/dev/null | head -1)
    [[ -n "$W" ]] && break
    sleep 5
done
if [[ -z "$W" ]]; then
    echo "[watchdog] no worker log under $LOGDIR after an hour; giving up"
    exit 0
fi

# Do not arm until traffic is flowing. The shm marker is not specific to a
# stall -- vLLM prints it during model load too, and in run 417809 it appeared 7
# times, the first 20 minutes before serving began. Dumping on that one gets a
# snapshot of a healthy loader and nothing else.
#
# Arm at the first benchmark phase rather than at profiling: 417809 wedged
# during WARMUP, so waiting for profiling would have missed it.
B="$LOGDIR/benchmark.out"
echo "[watchdog] waiting for the benchmark to start sending"
for _ in $(seq 1 1440); do
    if [[ -f "$B" ]] && grep -aq 'Phase \(warmup\|profiling\)' "$B"; then
        break
    fi
    sleep 5
done
if ! grep -aq 'Phase \(warmup\|profiling\)' "$B" 2>/dev/null; then
    echo "[watchdog] no benchmark phase within two hours; nothing to watch"
    exit 0
fi
echo "[watchdog] armed at $(date -u +%FT%TZ), tailing $W"

# Two dumps, 45 s apart: one stack is a photograph, two tell you whether it is
# stuck or merely slow.
n=0
tail -n0 -F "$W" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        *"$MARKER"*|*"$MARKER2"*)
            n=$((n + 1))
            [[ "$n" -gt 1 ]] && continue
            echo "[watchdog] stall marker seen at $(date -u +%FT%TZ)"
            dump_everything first
            sleep 45
            dump_everything second
            echo "[watchdog] done; exiting"
            exit 0
            ;;
    esac
done

exit 0
