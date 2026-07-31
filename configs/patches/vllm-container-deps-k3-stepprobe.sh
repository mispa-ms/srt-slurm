#!/usr/bin/env bash
# Kimi-K3 + step probe: count decode steps and how much text each one produced.
# =============================================================================
# WHY. On the agentic corpus the AgentX interactivity metric reports per-user
# token rates ~11x the maximum generation throughput the engine logs for itself
# (c1 run 60215671: engine max 357.1 tok/s, metric median 4,031 tok/s, 136/187
# records over). ITL is (last content arrival - first content arrival) /
# (OSL - 1); the Rust frontend drops a chunk entirely when a step's delta carries
# no text (chat_completions.rs:580 take_chunk), so the numerator can miss steps
# that the denominator still counts.
#
# That explanation has never been observed. The step count used so far was
# inferred as OSL / acceptance_length, which is an estimate, and no step that
# produced no text has ever actually been seen. This patch measures it instead.
#
# WHAT. Wraps BaseIncrementalDetokenizer.update, which is where a step's
# new_token_ids are turned into text (detokenizer.py:96, called once per engine
# step per request from output_processor.py:654). For every call it records the
# number of tokens committed and the number of characters the detokenizer
# appended, then logs a histogram every 30 s and once at exit:
#
#   STEPPROBE steps=N tokens=N chars=N silent_steps=N (pct) tok/step=... chars/step=...
#
# silent_steps counts calls that appended zero characters. If the explanation is
# right, that share is large on the speculative arms and near zero without
# speculation. If it is small, the explanation is wrong and the arrival gap comes
# from somewhere else.
#
# Read-only: it does not change tokens, text, stop handling or the return value.
# Appended to the module so it runs after the classes exist; if the anchor is
# missing the patch is skipped loudly rather than half-applied.
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps-k3-hfshim.sh ]]; then
    bash /configs/patches/vllm-container-deps-k3-hfshim.sh
fi

echo "=== k3-stepprobe: instrumenting the detokenizer ==="

TARGET="$(python3 -c 'import vllm.v1.engine.detokenizer as m; print(m.__file__)')"
echo "[stepprobe] target: $TARGET"

if ! grep -q "class BaseIncrementalDetokenizer" "$TARGET"; then
    echo "[stepprobe] FATAL: BaseIncrementalDetokenizer not found in $TARGET" >&2
    exit 1
fi
if grep -q "STEPPROBE" "$TARGET"; then
    echo "[stepprobe] already applied; skipping"
    exit 0
fi

cat >> "$TARGET" <<'PROBE'


# --- STEPPROBE (diagnostic; appended by vllm-container-deps-k3-stepprobe.sh) ---
# Counts decode steps and how many characters each one appended, to test whether
# steps that produce no text are what makes AgentX interactivity unreadable.
import atexit as _sp_atexit
import os as _sp_os
import threading as _sp_threading
import time as _sp_time


class _StepProbe:
    def __init__(self):
        self.lock = _sp_threading.Lock()
        self.steps = 0
        self.silent = 0
        self.tokens = 0
        self.chars = 0
        self.tok_hist = {}
        self.last = _sp_time.time()
        self.interval = float(_sp_os.environ.get("STEPPROBE_INTERVAL_S", "30"))

    def record(self, n_tokens, n_chars):
        with self.lock:
            self.steps += 1
            self.tokens += n_tokens
            self.chars += n_chars
            if n_chars == 0:
                self.silent += 1
            key = n_tokens if n_tokens <= 16 else 17
            self.tok_hist[key] = self.tok_hist.get(key, 0) + 1
            due = (_sp_time.time() - self.last) >= self.interval
            if due:
                self.last = _sp_time.time()
        if due:
            self.report()

    def report(self):
        with self.lock:
            if not self.steps:
                return
            hist = " ".join(
                f"{k if k <= 16 else '17+'}:{v}" for k, v in sorted(self.tok_hist.items())
            )
            print(
                f"STEPPROBE steps={self.steps} tokens={self.tokens} chars={self.chars} "
                f"silent_steps={self.silent} ({100.0 * self.silent / self.steps:.1f}%) "
                f"tok_per_step={self.tokens / self.steps:.2f} "
                f"chars_per_step={self.chars / self.steps:.2f} "
                f"chars_per_token={(self.chars / self.tokens) if self.tokens else 0:.2f} "
                f"tokens_per_step_hist[{hist}]",
                flush=True,
            )


_STEP_PROBE = _StepProbe()
_sp_atexit.register(_STEP_PROBE.report)

_sp_orig_update = BaseIncrementalDetokenizer.update


def _sp_update(self, new_token_ids, stop_terminated):
    before = len(self.output_text)
    result = _sp_orig_update(self, new_token_ids, stop_terminated)
    _STEP_PROBE.record(len(new_token_ids), len(self.output_text) - before)
    return result


BaseIncrementalDetokenizer.update = _sp_update
print("STEPPROBE armed on BaseIncrementalDetokenizer.update", flush=True)
# --- end STEPPROBE ---
PROBE

python3 -c 'import vllm.v1.engine.detokenizer' >/dev/null
echo "[stepprobe] applied and imports cleanly"
echo "=== k3-stepprobe: done ==="
