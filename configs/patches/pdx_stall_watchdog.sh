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
MARKER="No available shared memory broadcast block found"
HOST=$(hostname)
STAMP=$LOGDIR/.pyspy-watchdog-$HOST.claimed

# One watchdog per node. The setup script runs in every container on the node
# (worker and frontend), and two of these tailing the same file would dump the
# same processes twice.
if ! (set -o noclobber; : > "$STAMP") 2>/dev/null; then
    exit 0
fi

exec >> "$LOGDIR/pyspy-watchdog-$HOST.log" 2>&1
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
    local tag=$1 out="$LOGDIR/pyspy-stall-$HOST-$tag.txt"
    {
        echo "=== $(date -u +%FT%TZ)  stall dump: $tag ==="
        echo
        echo "--- processes ---"
        ps -eo pid,ppid,stat,etime,pcpu,comm,args --sort=pid | grep -aE "VllmWorker|EngineCore|python3" | grep -av grep
        echo
        for pid in $(pgrep -f 'VllmWorker|EngineCore' 2>/dev/null); do
            echo "--- py-spy dump --native --locals pid=$pid ---"
            timeout 60 py-spy dump --pid "$pid" --native --locals 2>&1 \
              || timeout 60 py-spy dump --pid "$pid" 2>&1 \
              || echo "  (py-spy failed for $pid)"
            echo
            echo "--- /proc/$pid/status wchan+state ---"
            grep -aE '^(State|Threads)' "/proc/$pid/status" 2>/dev/null
            echo
        done
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
echo "[watchdog] tailing $W"

# Two dumps, 45 s apart: one stack is a photograph, two tell you whether it is
# stuck or merely slow.
n=0
tail -n0 -F "$W" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        *"$MARKER"*)
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
