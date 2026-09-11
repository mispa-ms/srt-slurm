#!/usr/bin/env bash
# 09-11 nightly, arm b: K3_OURS=ssm. See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
