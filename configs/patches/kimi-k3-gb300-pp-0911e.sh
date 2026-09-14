#!/usr/bin/env bash
# 09-11 nightly, arm e: K3_OURS=ssm,evict -- the ssm chain plus the upstream
# engine-eviction fix, for arms with more than one prefill worker. Without it
# the decode engine dies on _cleanup_remote_engine as soon as one producer
# idles past engine_ttl (pipeline 67705738, both 2P1D arms).
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,evict"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
