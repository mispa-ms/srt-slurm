#!/bin/bash

set -euo pipefail

ISL="${PROFILE_ISL:-131072}"
OSL="${PROFILE_OSL:-1024}"
CACHE_FILL_OSL="${PROFILE_CACHE_FILL_OSL:-1}"
CONCURRENCY="${PROFILE_CONCURRENCY:-64}"
NUM_PROMPTS="${PROFILE_NUM_PROMPTS:-${CONCURRENCY}}"
SEED="${PROFILE_SEED:-0}"
MODEL_NAME="${PROFILE_MODEL_NAME:-moonshotai/Kimi-K3}"
TOKENIZER_PATH="${PROFILE_TOKENIZER_PATH:-/model}"
ENDPOINT="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}"
BENCHMARK_SCRIPT="/configs/kimi-wideep-benchmark-serving.py"

# Upstream resolved the client interpreter to a personal virtualenv on Lustre.
# That venv is a live source checkout on another cluster: it changes underneath
# a run, and it does not exist on ours. Take the container's python and let
# PROFILE_REPO_VENV opt back in, rather than requiring a path nobody else has.
REPO_VENV="${PROFILE_REPO_VENV:-}"
if [[ -n "${REPO_VENV}" ]]; then
    PYTHON_BIN="${REPO_VENV}/bin/python"
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "PROFILE_REPO_VENV set but ${PYTHON_BIN} is not executable" >&2
        exit 1
    fi
    unset VIRTUAL_ENV
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Error: ${PYTHON_BIN} not found; set PROFILE_REPO_VENV or PYTHON_BIN" >&2
        exit 127
    fi
fi

# benchmark_serving.py's full dependency set, imported before the run spends
# minutes building 128K-token prompts. `datasets` is deliberately absent: the
# wrapper stubs it so a non-random dataset fails loudly instead of importing.
# Falling back to a venv rather than failing follows sa-bench's own contract for
# containers that ship only a subset.
SA_BENCH_VENV="${SA_BENCH_VENV:-/tmp/sa-bench-venv}"
SA_BENCH_DEPS=(aiohttp numpy pandas Pillow tqdm transformers huggingface_hub)
if ! "${PYTHON_BIN}" -c "import aiohttp, numpy, pandas, PIL, tqdm, transformers, huggingface_hub" 2>/dev/null; then
    echo "Missing benchmark dependencies; installing into ${SA_BENCH_VENV} ..."
    if [[ ! -d "${SA_BENCH_VENV}" ]]; then
        "${PYTHON_BIN}" -m venv --system-site-packages "${SA_BENCH_VENV}"
    fi
    PYTHON_BIN="${SA_BENCH_VENV}/bin/python3"
    "${PYTHON_BIN}" -m pip install "${SA_BENCH_DEPS[@]}"
fi
if [[ "${PROFILE_VALIDATE_CLIENT_ONLY:-0}" == "1" ]]; then
    "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}" --help >/dev/null
    echo "CLIENT_VALIDATION_OK"
    exit 0
fi

source /srtctl-benchmarks/lib/profiling.sh
profiling_init_from_env

cleanup() {
    stop_all_profiling
    if [[ -n "${replay_pid:-}" ]] && kill -0 "${replay_pid}" 2>/dev/null; then
        kill "${replay_pid}" 2>/dev/null || true
        wait "${replay_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

run_phase() {
    local output_len="$1"
    local result_file="${2:-}"
    local -a save_args=()
    if [[ -n "${result_file}" ]]; then
        save_args=(
            --save-result
            --result-dir /logs/profile-benchmark
            --result-filename "${result_file}"
        )
    fi

    "${PYTHON_BIN}" -u "${BENCHMARK_SCRIPT}" \
        --model "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --base-url "${ENDPOINT}" \
        --backend dynamo \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len "${ISL}" \
        --random-output-len "${output_len}" \
        --random-range-ratio 1.0 \
        --random-num-workers 8 \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency "${CONCURRENCY}" \
        --request-rate inf \
        --seed "${SEED}" \
        --ignore-eos \
        --disable-tqdm \
        --trust-remote-code \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        "${save_args[@]}"
}

wait_for_full_decode_concurrency() {
    local marker="$1"
    local replay_pid="$2"
    local deadline=$((SECONDS + 600))

    while kill -0 "${replay_pid}" 2>/dev/null; do
        if [[ -e "${marker}" ]]; then
            echo "Decode-only profile window reached: ${CONCURRENCY} active streams"
            return 0
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for ${CONCURRENCY} active decode streams" >&2
            return 1
        fi
        sleep 0.25
    done

    echo "Replay exited before reaching ${CONCURRENCY} active decode streams" >&2
    return 1
}

mkdir -p /logs/profile-benchmark

# --- Does the prefix cache even fit the working set? -------------------------
# A two-wave replay is only a decode measurement if wave 2 hits. Wave 1 seeds
# NUM_PROMPTS x ISL tokens; if the prefill pool is smaller than that they are
# evicted before the replay and wave 2 silently re-prefills. That is not
# hypothetical: a prefill TP8/DCP2 arm ran with a 5,982,708-token pool against
# an 8,388,608-token working set, replayed at a ~22% hit, and its TTFT p90 went
# 10.2 s -> 46.2 s. Nothing failed; the numbers were just measuring the wrong
# thing. Cost of finding out the slow way: one four-hour job.
#
# The worker prints the pool at startup, so this is answerable before the first
# request. Advisory unless PROFILE_REQUIRE_KV_FIT=1, because the pool line is a
# log format we do not own.
WORKING_SET=$(( NUM_PROMPTS * ISL ))
# Only pass files that exist. A glob with no match stays literal, grep exits 2
# because it cannot open it, and under `set -euo pipefail` that kills the run --
# which is what this check did to the first disagg PP arm, silently, because the
# 2>/dev/null that was meant to quiet "no such file" also hid the cause. An
# advisory check must not be able to fail the job it is advising.
_kv_logs=()
for _pat in /logs/*prefill*.out /logs/*prefill*.out.log /logs/*agg*.out /logs/*agg*.out.log; do
    [[ -f "${_pat}" ]] && _kv_logs+=("${_pat}")
done
KV_POOL=""
if (( ${#_kv_logs[@]} > 0 )); then
    KV_POOL=$(grep -ohE "GPU KV cache size: [0-9,]+ tokens" "${_kv_logs[@]}" 2>/dev/null \
        | head -1 | grep -oE "[0-9,]+" | tr -d , || true)
fi
if [[ -z "${KV_POOL}" ]]; then
    echo "KV-fit check: could not read 'GPU KV cache size' from /logs; skipping."
else
    echo "KV-fit check: pool ${KV_POOL} tokens vs working set ${WORKING_SET} (${NUM_PROMPTS} x ${ISL})"
    if (( KV_POOL < WORKING_SET )); then
        echo "KV-fit check: POOL IS SMALLER THAN THE WORKING SET -- wave 2 will miss and this run will measure re-prefill, not decode." >&2
        if [[ "${PROFILE_REQUIRE_KV_FIT:-0}" == "1" ]]; then
            echo "KV-fit check: PROFILE_REQUIRE_KV_FIT=1, refusing to run." >&2
            exit 1
        fi
        echo "KV-fit check: continuing anyway (set PROFILE_REQUIRE_KV_FIT=1 to make this fatal)." >&2
    fi
fi

echo "Cache fill: ${NUM_PROMPTS} prompts, ISL=${ISL}, OSL=${CACHE_FILL_OSL}, concurrency=${CONCURRENCY}"
run_phase "${CACHE_FILL_OSL}"

echo "Cache fill complete; launching the identical-prompt replay"
echo "Profile replay: ${NUM_PROMPTS} prompts, ISL=${ISL}, OSL=${OSL}, concurrency=${CONCURRENCY}"
# The concurrency window exists to time start_all_profiling: a decode-only
# capture is only decode-only while every stream is generating. With profiling
# off there is nothing to time, and waiting for a marker no one writes turns a
# finished benchmark into a failed job -- which is what it did to the decode-PP
# DSpark arm (63781035). That arm completed 64/64 in both waves and was killed
# afterwards, by this, for never holding all 64 streams at once. Gate on the
# same predicate start_all_profiling uses, so the two cannot disagree.
if profiling__is_enabled; then
    decode_marker="/logs/profile-benchmark/decode-active-c${CONCURRENCY}"
    rm -f "${decode_marker}"
    export SRT_BENCH_DECODE_ACTIVE_MARKER="${decode_marker}"
    export SRT_BENCH_DECODE_ACTIVE_TARGET="${CONCURRENCY}"
    run_phase "${OSL}" "results_isl${ISL}_osl${OSL}_c${CONCURRENCY}.json" &
    replay_pid=$!
    unset SRT_BENCH_DECODE_ACTIVE_MARKER SRT_BENCH_DECODE_ACTIVE_TARGET
    # Still fatal here: a capture taken outside the window measures the wrong
    # thing, and that is worse than no capture.
    wait_for_full_decode_concurrency "${decode_marker}" "${replay_pid}"
    start_all_profiling
    wait "${replay_pid}"
else
    run_phase "${OSL}" "results_isl${ISL}_osl${OSL}_c${CONCURRENCY}.json"
fi

stop_all_profiling
trap - EXIT
