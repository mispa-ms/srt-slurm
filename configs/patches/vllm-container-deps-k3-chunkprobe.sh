#!/usr/bin/env bash
# Kimi-K3 + Rust probes: count tokens that produce no text, and chunks not sent.
# =============================================================================
# WHY. On the agentic corpus the AgentX interactivity metric reports per-user
# token rates ~11x the peak generation throughput the engine logs for itself
# (c1 run 60215671: server peak 357.1 tok/s, metric median 4,031 tok/s, 136/187
# records above the server's own peak). ITL is
# (last content arrival - first content arrival) / (OSL - 1), and take_chunk
# drops a chunk entirely when a step's delta carries no text, so steps that
# produce nothing leave the numerator while their tokens stay in the
# denominator. Every part of that has been inferred; none of it observed.
#
# A first attempt instrumented the Python detokenizer and logged nothing,
# because with VLLM_USE_RUST_FRONTEND=1 detokenization is Rust too
# (rust/src/tokenizer/src/incremental.rs). Both measurements therefore live here.
#
# WHAT IS COUNTED
#   DECODEPROBE  push_token (incremental.rs) returns how many string bytes a
#                token added. Its early `return Ok(0)` is the token-produced-no-
#                text path -- the string did not grow, or it ends in U+FFFD, an
#                incomplete UTF-8 sequence. Counters on both return paths give
#                tokens, bytes, and the share that yielded nothing.
#   CHUNKPROBE   take_chunk (chat_completions.rs) returns None when a delta has
#                no content, reasoning or tool_calls. Counters on that early
#                return and on the emit path give suppressed vs emitted chunks.
#
# Together: how many tokens produce no text, and how many steps therefore send
# no chunk. If both are large on the speculative arm and near zero on the
# control, the explanation holds and nothing in it is inferred any more.
#
# SOURCE. The image is vllm/vllm-openai:kimi-k3, whose commit is NOT on
# vllm-project/vllm main -- K3 day-0 support lives on the `kimi-k3` branch (PR
# #50000 unmerged), so fetching the sha from the image fails with "couldn't find
# remote ref". This fetches the branch instead and reports whether the image's
# commit is an ancestor. The anchor checks below are the real safety net: a
# source/image mismatch that moved either patch site aborts the build.
#
# Read-only: tokens, text, chunk contents and every return value are unchanged.
#
# Build cost is why there is a cache: the workspace is large and a cold build is
# tens of minutes. Layout and the flock-on-FD-201 pattern follow the dynamo
# source-install cache from NVIDIA/srt-slurm#280.
#
# The recipe must point the frontend at the result:
#     VLLM_RUST_FRONTEND_PATH: /configs/vllm-rust/chunkprobe/vllm-rs
# =============================================================================
set -euo pipefail

if [[ -f /configs/patches/vllm-container-deps-k3-hfshim.sh ]]; then
    bash /configs/patches/vllm-container-deps-k3-hfshim.sh
fi

echo "=== k3-chunkprobe: patched Rust frontend ==="

CACHE_ROOT="/configs/vllm-rust"
OUT_DIR="$CACHE_ROOT/chunkprobe"
LOCK="$CACHE_ROOT/.chunkprobe.lock"
VLLM_BRANCH="${CHUNKPROBE_VLLM_BRANCH:-kimi-k3}"
mkdir -p "$CACHE_ROOT"

VLLM_VER="$(python3 -c 'import vllm; print(vllm.__version__)')"
VLLM_SHA="$(printf '%s' "$VLLM_VER" | sed -n 's/.*+g\([0-9a-f]\{7,\}\).*/\1/p')"
echo "[chunkprobe] image vLLM $VLLM_VER (commit ${VLLM_SHA:-unknown}), building from branch $VLLM_BRANCH"

(
flock -x 201
KEY="$VLLM_BRANCH-${VLLM_SHA:-unknown}"
if [[ -f "$OUT_DIR/.complete" ]] && [[ "$(cat "$OUT_DIR/.build-key" 2>/dev/null)" == "$KEY" ]]; then
    echo "[chunkprobe] cache hit for $KEY"
else
    echo "[chunkprobe] cold build for $KEY"
    rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"
    apt-get update -qq && apt-get install -y -qq git curl build-essential pkg-config libssl-dev protobuf-compiler libclang-dev > /dev/null 2>&1
    if ! command -v cargo >/dev/null 2>&1; then
        curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal > /dev/null
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    SRC=/tmp/vllm-src; rm -rf "$SRC"; mkdir -p "$SRC"; cd "$SRC"
    git init -q && git remote add origin https://github.com/vllm-project/vllm.git
    git fetch -q --depth 200 origin "$VLLM_BRANCH"
    git checkout -q FETCH_HEAD
    echo "[chunkprobe] branch $VLLM_BRANCH is at $(git rev-parse --short HEAD)"
    if [[ -n "$VLLM_SHA" ]]; then
        if git merge-base --is-ancestor "$VLLM_SHA" HEAD 2>/dev/null; then
            echo "[chunkprobe] image commit $VLLM_SHA is an ancestor of the branch tip"
        else
            echo "[chunkprobe] NOTE: image commit $VLLM_SHA is not reachable in the fetched history."
            echo "[chunkprobe] Building the branch tip instead; the anchor checks below decide whether that is safe."
        fi
    fi

    python3 - <<'PY'
import sys

DEC = "rust/src/tokenizer/src/incremental.rs"
CHAT = "rust/src/server/src/routes/openai/chat_completions.rs"

# --- DECODEPROBE: a token that adds no string bytes takes the early return ---
dec = open(DEC).read()
a1 = """        if string.len() <= prefix_len || string.ends_with('\\u{FFFD}') {
            return Ok(0);
        }"""
a2 = """        Ok(new_chunk.len())
    }"""
for anchor, name in ((a1, "push_token early return"), (a2, "push_token tail")):
    if anchor not in dec:
        sys.exit(f"[chunkprobe] FATAL: {name} anchor not found in {DEC}; refusing a partial patch")
dec = dec.replace(a1, """        if string.len() <= prefix_len || string.ends_with('\\u{FFFD}') {
            DECODEPROBE_TOKENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let s = DECODEPROBE_SILENT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if s % 5000 == 0 {
                tracing::info!(
                    "DECODEPROBE tokens={} silent={} bytes={}",
                    DECODEPROBE_TOKENS.load(std::sync::atomic::Ordering::Relaxed),
                    s,
                    DECODEPROBE_BYTES.load(std::sync::atomic::Ordering::Relaxed)
                );
            }
            return Ok(0);
        }""", 1)
dec = dec.replace(a2, """        let produced = new_chunk.len();
        DECODEPROBE_TOKENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DECODEPROBE_BYTES.fetch_add(produced as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(produced)
    }""", 1)
dec += """

// --- DECODEPROBE (diagnostic; injected by vllm-container-deps-k3-chunkprobe.sh) ---
// tokens: every token pushed. silent: those that added no string bytes, i.e. the
// string did not grow or ended in U+FFFD. bytes: total string bytes produced.
static DECODEPROBE_TOKENS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODEPROBE_SILENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODEPROBE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// --- end DECODEPROBE ---
"""
open(DEC, "w").write(dec)
print("[chunkprobe] DECODEPROBE applied to", DEC)

# --- CHUNKPROBE: a delta with no content/reasoning/tool_calls sends no chunk ---
chat = open(CHAT).read()
a3 = """        if !has_delta && logprobs.is_none() && token_ids.is_none() {
            return None;
        }"""
if a3 not in chat:
    sys.exit(f"[chunkprobe] FATAL: take_chunk early-return anchor not found in {CHAT}")
chat = chat.replace(a3, """        if !has_delta && logprobs.is_none() && token_ids.is_none() {
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
        CHUNKPROBE_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);""", 1)
chat += """

// --- CHUNKPROBE (diagnostic; injected by vllm-container-deps-k3-chunkprobe.sh) ---
// suppressed: steps whose delta carried nothing, so no SSE chunk was sent. Those
// steps leave ITL's numerator while their tokens stay in its denominator.
static CHUNKPROBE_SUPPRESSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CHUNKPROBE_EMITTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// --- end CHUNKPROBE ---
"""
open(CHAT, "w").write(chat)
print("[chunkprobe] CHUNKPROBE applied to", CHAT)
PY

    cd "$SRC/rust"
    cargo build --release --bin vllm-rs 2>&1 | tail -8
    BIN="$SRC/rust/target/release/vllm-rs"
    test -x "$BIN" || { echo "[chunkprobe] FATAL: build produced no vllm-rs" >&2; exit 1; }
    cp "$BIN" "$OUT_DIR/vllm-rs"
    printf '%s' "$KEY" > "$OUT_DIR/.build-key"
    touch "$OUT_DIR/.complete"
    echo "[chunkprobe] built -> $OUT_DIR/vllm-rs"
fi
) 201>"$LOCK"

test -x "$OUT_DIR/vllm-rs" || { echo "[chunkprobe] FATAL: $OUT_DIR/vllm-rs missing after build" >&2; exit 1; }
echo "[chunkprobe] ready: $OUT_DIR/vllm-rs"
echo "=== k3-chunkprobe: done ==="
