#!/usr/bin/env bash
# Arm b-norev of the 09-08 matrix: K3_OURS=ssm, and the vllm#53774 (#52388 revert) step skipped.
export K3_OURS="ssm"
export K3_SKIP_REVERT52388=1
exec bash /configs/patches/kimi-k3-gb300-pp-908.sh
