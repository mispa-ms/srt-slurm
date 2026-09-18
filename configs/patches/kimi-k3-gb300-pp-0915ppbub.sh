#!/usr/bin/env bash
# 0915a plus ppbub only: each PP rank logs its step cadence and the time it
# spends blocked waiting for the previous stage's tensors, so the pipeline
# bubble is measured rather than inferred from a throughput ratio.
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mcclamp,mcpp,mrcap,pushdone,ppbub"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
