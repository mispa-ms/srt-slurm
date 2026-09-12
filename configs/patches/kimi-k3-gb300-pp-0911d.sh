#!/usr/bin/env bash
# 09-11 nightly, arm d: K3_OURS=ssm,dpp -- adds pipeline-parallel decode.
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,dpp"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
