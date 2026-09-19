#!/usr/bin/env bash
# 0915a plus kvgroup: the KV-cache group size is chosen to minimise padding
# when VLLM_KV_GROUP_MIN_PADDING=1, which the arm sets on BOTH roles. Inert
# without that variable.
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mcclamp,mcpp,mrcap,pushdone,kvgroup"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
