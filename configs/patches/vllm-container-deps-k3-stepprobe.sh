#!/usr/bin/env bash
# Kimi-K3 + engine-step probe: count decode steps and tokens per request.
# =============================================================================
# WHY. On the agentic corpus the AgentX interactivity metric reports per-user
# token rates ~11x the peak generation throughput the engine logs for itself
# (c1 run 60215671: server peak 357.1 tok/s, metric median 4,031, 136/187
# records above the peak). ITL is
# (last content arrival - first content arrival) / (OSL - 1), and take_chunk
# (chat_completions.rs) sends no SSE chunk when a step's delta carries no text,
# so steps that produce nothing leave the numerator while their tokens stay in
# the denominator.
#
# The step count in that argument has always been inferred as
# OSL / acceptance_length. This measures it.
#
# WHY HERE. A step emits at most one chunk, so comparing steps against the
# client's arrival count settles the question on its own:
#     steps ~= arrivals  -> one chunk per step, and the tokens-per-step needed
#                           to reach OSL contradicts the logged acceptance length
#     steps >> arrivals  -> most steps sent no chunk; suppression is real
#
# The first attempt instrumented the Python detokenizer and logged nothing:
# with VLLM_USE_RUST_FRONTEND=1 detokenization is Rust
# (rust/src/tokenizer/src/incremental.rs). Scheduling and sampling are not --
# EngineCore.step() (v1/engine/core.py) runs in the Python engine process
# whatever the frontend is, and its EngineCoreOutput.new_token_ids are the
# tokens that step committed. Rebuilding the Rust side is not an option here
# anyway: the image's commit is not published on vllm-project/vllm.
#
# WHAT IT LOGS. Per request, once it finishes:
#     ENGINEPROBE req=<id> steps=N tokens=N tok_per_step=...
# and a rollup every 30 s:
#     ENGINEPROBE_TOTAL engine_steps=N reqs_done=N steps_per_req=... tok_per_step=...
#
# Read-only: outputs are returned untouched.
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps-k3-hfshim.sh ]]; then
    bash /configs/patches/vllm-container-deps-k3-hfshim.sh
fi

echo "=== k3-stepprobe: instrumenting EngineCore.step ==="

TARGET="$(python3 -c 'import vllm.v1.engine.core as m; print(m.__file__)')"
echo "[stepprobe] target: $TARGET"
grep -q "class EngineCore" "$TARGET" || { echo "[stepprobe] FATAL: EngineCore not found in $TARGET" >&2; exit 1; }
if grep -q "ENGINEPROBE" "$TARGET"; then echo "[stepprobe] already applied"; exit 0; fi

cat >> "$TARGET" <<'PROBE'


# --- ENGINEPROBE (diagnostic; appended by vllm-container-deps-k3-stepprobe.sh) ---
# Counts engine steps and the tokens each one commits per request, so the step
# count stops being inferred from OSL / acceptance_length. A step emits at most
# one SSE chunk, so steps-per-request against the client's arrival count shows
# directly whether chunks are being suppressed.
import threading as _ep_threading
import time as _ep_time


class _EngineProbe:
    def __init__(self):
        self.lock = _ep_threading.Lock()
        self.per_req = {}
        self.engine_steps = 0
        self.reqs_done = 0
        self.done_steps = 0
        self.done_tokens = 0
        self.last = _ep_time.time()

    def observe(self, outputs):
        lines = []
        with self.lock:
            self.engine_steps += 1
            for core_outputs in outputs.values():
                for out in getattr(core_outputs, "outputs", ()) or ():
                    n = len(getattr(out, "new_token_ids", ()) or ())
                    slot = self.per_req.setdefault(out.request_id, [0, 0])
                    if n:
                        slot[0] += 1
                        slot[1] += n
                    if getattr(out, "finish_reason", None) is not None:
                        steps, tokens = self.per_req.pop(out.request_id, (0, 0))
                        self.reqs_done += 1
                        self.done_steps += steps
                        self.done_tokens += tokens
                        lines.append(
                            f"ENGINEPROBE req={out.request_id} steps={steps} "
                            f"tokens={tokens} "
                            f"tok_per_step={(tokens / steps) if steps else 0:.2f}"
                        )
            due = (_ep_time.time() - self.last) >= 30.0
            if due:
                self.last = _ep_time.time()
                lines.append(
                    f"ENGINEPROBE_TOTAL engine_steps={self.engine_steps} "
                    f"reqs_done={self.reqs_done} "
                    f"steps_per_req={(self.done_steps / self.reqs_done) if self.reqs_done else 0:.1f} "
                    f"tok_per_step={(self.done_tokens / self.done_steps) if self.done_steps else 0:.2f} "
                    f"in_flight={len(self.per_req)}"
                )
        for line in lines:
            print(line, flush=True)


_ENGINE_PROBE = _EngineProbe()
_ep_orig_step = EngineCore.step


def _ep_step(self):
    outputs, executed = _ep_orig_step(self)
    try:
        if outputs:
            _ENGINE_PROBE.observe(outputs)
    except Exception:
        pass
    return outputs, executed


EngineCore.step = _ep_step

if hasattr(EngineCore, "step_with_batch_queue"):
    _ep_orig_step_bq = EngineCore.step_with_batch_queue

    def _ep_step_bq(self):
        outputs, executed = _ep_orig_step_bq(self)
        try:
            if outputs:
                _ENGINE_PROBE.observe(outputs)
        except Exception:
            pass
        return outputs, executed

    EngineCore.step_with_batch_queue = _ep_step_bq

print("ENGINEPROBE armed on EngineCore.step", flush=True)
# --- end ENGINEPROBE ---
PROBE

python3 -c 'import vllm.v1.engine.core'
echo "=== k3-stepprobe: done ==="
