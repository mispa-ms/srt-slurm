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

bash /configs/patches/vllm-container-deps-k3-b200-dspark-pp-main.sh
bash /configs/patches/vllm-container-deps-k3-stackdump-install.sh
bash /configs/patches/vllm-container-deps-k3-warmupdiag-install.sh
