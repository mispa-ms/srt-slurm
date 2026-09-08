#!/usr/bin/env bash
# Arm b of the 09-08 A/B: K3_OURS=ssm. See kimi-k3-gb300-pp-908.sh.
export K3_OURS="ssm"
exec bash /configs/patches/kimi-k3-gb300-pp-908.sh
