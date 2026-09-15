#!/usr/bin/env bash
# 09-12 nightly, arm b: K3_OURS=ssm,mcclamp,mcpp,mrcap -- 0914a plus the
# MooncakeStore PP handshake override.
#
# Needed by any arm that runs a mooncake tier AND a pipeline stage. Without it
# the store refuses the peer outright:
#   ValueError: MooncakeStoreConnector received pp_rank > 0 handshake metadata
#               but does not support PP-disaggregated KV transfer.
# Both -sarepropp2 arms of 67894753 died there; 0914a omitted mcpp because the
# arms it was written for had no tier.
#
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mcclamp,mcpp,mrcap"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
