#!/usr/bin/env bash
# 09-11 nightly, arm j: K3_OURS=ssm,mrcap -- arm b's chain plus the NIXL
# memory-region cap. 1P2D runs one prefill worker, so it does not need evict.
# The knob defaults to off, which makes this chain arm b byte for byte; the arm
# that sets VLLM_NIXL_MAX_MR_BYTES is the one variable.
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mrcap"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
