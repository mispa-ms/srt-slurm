#!/usr/bin/env bash
# Instrument warmup_kernels so the two PP stages can be compared step for step.
# =============================================================================
# WHAT THE STACKS ALREADY GAVE US. The ns7 PP2 deadlock on the current nightly is
# not in capture_model, which is where this workstream looked first. Both stages
# get through capture. Then:
#
#   stage 0   gpu_worker.py:821  compile_or_warm_up_model
#             warmup.py:319      warmup_kernels   <- worker_execute_model(prefill_output)
#             gpu_worker.py:1146 execute_model
#             parallel_state.py:1051 isend_tensor_dict -> send_object   BLOCKED
#
#   stage 1   multiproc_executor.py:1033 worker_busy_loop -> shm_broadcast recv
#
# warmup.py:319 is the *first* execute_model in warmup_kernels, the prefill. So
# stage 0 is trying to hand the first warmup batch across the pipeline and stage 1
# has already returned from the whole compile_or_warm_up_model RPC and gone back to
# waiting for work. The two stages disagree about whether warmup does a PP exchange
# at all.
#
# WHAT IS ALREADY RULED OUT. use_workspace_lane is a ContextVar, not a collective.
# Memory is not tight -- empty_cache leaves 75 of 178 GiB. use_v2_model_runner comes
# from VLLM_USE_V2_MODEL_RUNNER, which these configs set to 1 for every rank, so both
# stages should take the warmup_kernels branch. num_speculative_steps is
# vllm_config.num_speculative_tokens, identical on both stages, so the decode-step
# count cannot differ. Our PP patch touches neither warmup.py nor gpu_worker.py.
#
# WHY NOT JUST TURN V2 OFF. That was the obvious A/B and it is invalid: config/vllm.py
# says DSpark is implemented only by the V2 runner and forces V2 whenever the
# speculative method is dspark, so VLLM_USE_V2_MODEL_RUNNER=0 either raises or
# silently drops speculation. The arm would answer a question we are not asking.
#
# WHAT THIS ADDS. Five numbers per rank at the top of warmup_kernels, one line as it
# enters the prefill, and one on the way out. Reading stage 0's against stage 1's
# says immediately whether stage 1 entered the function, how many steps each intended
# to run, and where stage 1 left. Chains the stack dump too, so if the numbers agree
# the stacks are still there to explain it.
# =============================================================================
set -euo pipefail


echo "=== warmupdiag: instrument warmup_kernels ==="

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
target = os.path.join(root, "vllm/v1/worker/gpu/warmup.py")
src = open(target).read()

if "[warmupdiag]" in src:
    print("[warmupdiag] already applied: " + target)
    sys.exit(0)

# 1. entry banner, right after the mm-encoder early return
ENTRY_ANCHOR = """    if model_runner.vllm_config.is_mm_encoder_only:
        return

    num_spec_steps = model_runner.num_speculative_steps
"""
if src.count(ENTRY_ANCHOR) != 1:
    sys.exit(
        "[warmupdiag] FATAL: expected one warmup_kernels entry anchor, found %d"
        % src.count(ENTRY_ANCHOR)
    )
ENTRY = ENTRY_ANCHOR + '''
    # [warmupdiag] Identity of this rank, before anything is derived from it.
    logger.info(
        "[warmupdiag] enter warmup_kernels: pp_rank=%s pp_size=%s is_last=%s "
        "use_v2=%s num_spec_steps=%s speculator=%s",
        get_pp_group().rank_in_group,
        get_pp_group().world_size,
        model_runner.is_last_pp_rank,
        model_runner.vllm_config.use_v2_model_runner,
        num_spec_steps,
        type(getattr(model_runner, "speculator", None)).__name__,
    )
'''

# 1b. the batch size, which is what the whole question turns on
SIZE_ANCHOR = """    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]
"""
if src.count(SIZE_ANCHOR) != 1:
    sys.exit(
        "[warmupdiag] FATAL: expected one warmup req_ids line, found %d"
        % src.count(SIZE_ANCHOR)
    )
SIZE = '''    # [warmupdiag] warmup_kernels sizes its batch from THIS RANK'S kv cache, and
    # the PP stages are not identical: the model splits [47,46] and their Mooncake
    # registrations differ (num_segments 12 vs 18). If num_reqs or the scheduled
    # token count differ across stages, the prefill send and recv disagree on shape
    # and the pipeline deadlocks exactly where the stacks put it.
    logger.info(
        "[warmupdiag] batch: pp_rank=%s num_reqs=%s max_blocks_per_req=%s "
        "num_blocks=%s prompt_len=%s decode_query_len=%s decode_len=%s "
        "num_decode_steps=%s max_num_seqs=%s max_num_batched=%s "
        "prefill_counts=%s decode_counts=%s groups=%s",
        get_pp_group().rank_in_group,
        num_reqs,
        max_blocks_per_req,
        model_runner.kv_cache_config.num_blocks,
        prompt_len,
        decode_query_len,
        decode_len,
        num_decode_steps,
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens,
        prefill_block_counts,
        decode_block_counts,
        num_kv_cache_groups,
    )
''' + SIZE_ANCHOR
src = src.replace(SIZE_ANCHOR, SIZE, 1)
src = src.replace(ENTRY_ANCHOR, ENTRY, 1)

# 2. one line immediately before the first (prefill) execute_model, and one after
PREFILL_ANCHOR = """    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)
"""
if src.count(PREFILL_ANCHOR) != 1:
    sys.exit(
        "[warmupdiag] FATAL: expected one warmup prefill call, found %d"
        % src.count(PREFILL_ANCHOR)
    )
src = src.replace(
    PREFILL_ANCHOR,
    """    model_runner.kv_connector.set_disabled(True)
    logger.info("[warmupdiag] prefill execute_model: entering")
    worker_execute_model(prefill_output)
    logger.info("[warmupdiag] prefill execute_model: returned")
""",
    1,
)

# 3. exit banner, at the single cleanup step that ends the function
EXIT_ANCHOR = """    worker_execute_model(cleanup_output)
    model_runner.kv_connector.set_disabled(False)
"""
if src.count(EXIT_ANCHOR) != 1:
    sys.exit(
        "[warmupdiag] FATAL: expected one warmup cleanup call, found %d"
        % src.count(EXIT_ANCHOR)
    )
src = src.replace(
    EXIT_ANCHOR,
    """    logger.info("[warmupdiag] cleanup execute_model: entering")
    worker_execute_model(cleanup_output)
    logger.info("[warmupdiag] leaving warmup_kernels")
    model_runner.kv_connector.set_disabled(False)
""",
    1,
)

# The file logs elsewhere, but do not assume: get_pp_group is what it may lack.
if "get_pp_group" not in src.split("[warmupdiag]")[0]:
    IMPORT_ANCHOR = "from vllm.logger import init_logger\n"
    if src.count(IMPORT_ANCHOR) != 1:
        sys.exit(
            "[warmupdiag] FATAL: no init_logger import to anchor the pp import to"
        )
    src = src.replace(
        IMPORT_ANCHOR,
        "from vllm.distributed.parallel_state import get_pp_group\n" + IMPORT_ANCHOR,
        1,
    )

compile(src, target, "exec")
open(target, "w").write(src)
print("[warmupdiag] applied: " + target)
PY

# The instrumentation is worthless if the module no longer imports, and a broken
# import would look exactly like the deadlock we are chasing.
python3 -c "
import vllm.v1.worker.gpu.warmup as w, inspect, sys
src = inspect.getsource(w.warmup_kernels)
n = src.count('[warmupdiag]')
print('[warmupdiag] verified: module imports, %d markers in warmup_kernels' % n)
sys.exit(0 if n >= 5 else 1)
"

echo "=== warmupdiag: done ==="
