#!/usr/bin/env bash
# 0915a plus mcppdbg: the mooncake store logs its lookup prefixes and save key
# per rank. Diagnostic arm for the PP2 store never being read.
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mcclamp,mcpp,mrcap,pushdone,mcppdbg"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
