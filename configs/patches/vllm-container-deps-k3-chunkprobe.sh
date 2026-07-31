#!/usr/bin/env bash
# Kimi-K3 + chunk probe: count SSE chunks the Rust frontend suppresses.
# =============================================================================
# WHY. take_chunk (rust/src/server/src/routes/openai/chat_completions.rs) returns
# None when a step's delta carries no content, reasoning or tool_calls, so that
# step sends no SSE chunk. ITL's numerator is the span between the first and last
# content arrival while its denominator is OSL - 1, so suppressed steps drop out
# of the numerator but keep their tokens in the denominator. That is the standing
# explanation for interactivity reading ~11x the engine's own peak throughput on
# the agentic corpus, and it has only ever been inferred.
#
# The companion probe (vllm-container-deps-k3-stepprobe.sh) measures the engine
# side: steps taken and characters produced. This one measures the wire side:
# how many of those steps actually became a chunk. Together they close the chain
# with observations instead of arithmetic.
#
# HOW. vLLM's Rust frontend is a standalone binary (`vllm-rs`, rust/src/cmd)
# located at run time through VLLM_RUST_FRONTEND_PATH, which defaults to "auto"
# (vllm/envs.py:158, entrypoints/cli/serve.py:61). So no srt-slurm schema change
# is needed: build a patched binary here and point the recipe's environment at
# it. The recipe must set
#     VLLM_RUST_FRONTEND_PATH: /configs/vllm-rust/chunkprobe/vllm-rs
#
# The patch adds two counters around take_chunk's early return and logs them
# every 1000 suppressions:
#     CHUNKPROBE suppressed=N emitted=N
#
# Build cost is the reason for the cache: the workspace is large, so a cold build
# is tens of minutes. Layout and the flock-on-FD-201 pattern follow the dynamo
# source-install cache added in NVIDIA/srt-slurm#280 -- one builder, everyone
# else waits on the lock and reads the sentinel. Keyed by the vLLM commit baked
# into the image, so a new image rebuilds rather than silently reusing.
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps-k3-hfshim.sh ]]; then
    bash /configs/patches/vllm-container-deps-k3-hfshim.sh
fi

echo "=== k3-chunkprobe: patched Rust frontend ==="

CACHE_ROOT="/configs/vllm-rust"
OUT_DIR="$CACHE_ROOT/chunkprobe"
LOCK="$CACHE_ROOT/.chunkprobe.lock"
mkdir -p "$CACHE_ROOT"

# The image's vLLM version carries the commit as the local part: 0.1.devN+g<sha>.
VLLM_VER="$(python3 -c 'import vllm; print(vllm.__version__)')"
VLLM_SHA="$(printf '%s' "$VLLM_VER" | sed -n 's/.*+g\([0-9a-f]\{7,\}\).*/\1/p')"
if [[ -z "$VLLM_SHA" ]]; then
    echo "[chunkprobe] FATAL: no commit in vllm.__version__='$VLLM_VER'; cannot match sources to the image." >&2
    exit 1
fi
echo "[chunkprobe] image vLLM $VLLM_VER -> commit $VLLM_SHA"

(
flock -x 201
if [[ -f "$OUT_DIR/.complete" ]] && [[ "$(cat "$OUT_DIR/.build-sha" 2>/dev/null)" == "$VLLM_SHA" ]]; then
    echo "[chunkprobe] cache hit for $VLLM_SHA"
else
    echo "[chunkprobe] cold build for $VLLM_SHA"
    rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"
    apt-get update -qq && apt-get install -y -qq git curl build-essential pkg-config libssl-dev protobuf-compiler libclang-dev > /dev/null 2>&1
    if ! command -v cargo >/dev/null 2>&1; then
        curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal > /dev/null
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    SRC=/tmp/vllm-src; rm -rf "$SRC"; mkdir -p "$SRC"; cd "$SRC"
    git init -q && git remote add origin https://github.com/vllm-project/vllm.git
    git fetch -q --depth 1 origin "$VLLM_SHA" && git checkout -q FETCH_HEAD

    F=rust/src/server/src/routes/openai/chat_completions.rs
    test -f "$F" || { echo "[chunkprobe] FATAL: $F missing at $VLLM_SHA" >&2; exit 1; }
    grep -q "fn take_chunk" "$F" || { echo "[chunkprobe] FATAL: take_chunk not found in $F" >&2; exit 1; }

    python3 - "$F" <<'PY'
import re, sys
path = sys.argv[1]
src = open(path).read()
anchor = """        if !has_delta && logprobs.is_none() && token_ids.is_none() {
            return None;
        }"""
if anchor not in src:
    sys.exit("[chunkprobe] FATAL: take_chunk early-return anchor not found; refusing a partial patch")
patched = """        if !has_delta && logprobs.is_none() && token_ids.is_none() {
            let n = CHUNKPROBE_SUPPRESSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 1000 == 0 {
                tracing::info!(
                    "CHUNKPROBE suppressed={} emitted={}",
                    n,
                    CHUNKPROBE_EMITTED.load(std::sync::atomic::Ordering::Relaxed)
                );
            }
            return None;
        }
        CHUNKPROBE_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);"""
src = src.replace(anchor, patched, 1)
src += """

// --- CHUNKPROBE (diagnostic; injected by vllm-container-deps-k3-chunkprobe.sh) ---
// Counts steps whose delta carried nothing, so no SSE chunk was sent. Those steps
// leave ITL's numerator while their tokens stay in its denominator.
static CHUNKPROBE_SUPPRESSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CHUNKPROBE_EMITTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// --- end CHUNKPROBE ---
"""
open(path, "w").write(src)
print("[chunkprobe] patch applied to", path)
PY

    cd "$SRC/rust"
    cargo build --release --bin vllm-rs 2>&1 | tail -5
    BIN="$SRC/rust/target/release/vllm-rs"
    test -x "$BIN" || { echo "[chunkprobe] FATAL: build produced no vllm-rs" >&2; exit 1; }
    cp "$BIN" "$OUT_DIR/vllm-rs"
    printf '%s' "$VLLM_SHA" > "$OUT_DIR/.build-sha"
    touch "$OUT_DIR/.complete"
    echo "[chunkprobe] built -> $OUT_DIR/vllm-rs"
fi
) 201>"$LOCK"

test -x "$OUT_DIR/vllm-rs" || { echo "[chunkprobe] FATAL: $OUT_DIR/vllm-rs missing after build" >&2; exit 1; }
echo "[chunkprobe] ready: $OUT_DIR/vllm-rs  (recipe must set VLLM_RUST_FRONTEND_PATH to this)"
echo "=== k3-chunkprobe: done ==="
