#!/usr/bin/env bash
# 0915a plus ppstep only: the engine core logs tokens-scheduled and step wall
# time per completed step, so prefill throughput can be split into its two
# factors instead of inferred from an assumed 16,384-token batch.
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mcclamp,mcpp,mrcap,pushdone,ppstep"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
