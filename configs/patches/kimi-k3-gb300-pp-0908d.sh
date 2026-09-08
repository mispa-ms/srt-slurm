#!/usr/bin/env bash
# Arm d of the 09-08 A/B: K3_OURS=ssm,mcpp. See kimi-k3-gb300-pp-908.sh.
export K3_OURS="ssm,mcpp"
exec bash /configs/patches/kimi-k3-gb300-pp-908.sh
