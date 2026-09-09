#!/usr/bin/env bash
# 09-09 nightly, arm b: K3_OURS=ssm. See kimi-k3-gb300-pp-909.sh.
export K3_OURS="ssm"
exec bash /configs/patches/kimi-k3-gb300-pp-909.sh
