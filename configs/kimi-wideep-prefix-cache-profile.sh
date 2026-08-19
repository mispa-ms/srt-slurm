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
REPO_VENV="${PROFILE_REPO_VENV:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weizha/vllm/.venv}"
PYTHON_BIN="${REPO_VENV}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Repository virtualenv is unavailable at ${REPO_VENV}" >&2
    exit 1
fi
unset VIRTUAL_ENV
if ! "${PYTHON_BIN}" -c 'import transformers'; then
    echo "Benchmark dependencies are unavailable to ${PYTHON_BIN}" >&2
    exit 1
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

echo "Cache fill: ${NUM_PROMPTS} prompts, ISL=${ISL}, OSL=${CACHE_FILL_OSL}, concurrency=${CONCURRENCY}"
run_phase "${CACHE_FILL_OSL}"

echo "Cache fill complete; launching the identical-prompt replay"
echo "Profile replay: ${NUM_PROMPTS} prompts, ISL=${ISL}, OSL=${OSL}, concurrency=${CONCURRENCY}"
decode_marker="/logs/profile-benchmark/decode-active-c${CONCURRENCY}"
rm -f "${decode_marker}"
export SRT_BENCH_DECODE_ACTIVE_MARKER="${decode_marker}"
export SRT_BENCH_DECODE_ACTIVE_TARGET="${CONCURRENCY}"
run_phase "${OSL}" "results_isl${ISL}_osl${OSL}_c${CONCURRENCY}.json" &
replay_pid=$!
unset SRT_BENCH_DECODE_ACTIVE_MARKER SRT_BENCH_DECODE_ACTIVE_TARGET
wait_for_full_decode_concurrency "${decode_marker}" "${replay_pid}"
start_all_profiling
wait "${replay_pid}"

stop_all_profiling
trap - EXIT
