#!/usr/bin/env bash
# 09-14 nightly, arm a: K3_OURS=ssm,mcclamp,mrcap -- the ssm chain on the
# rebased #50494/#50499 head bc2e6269e4, plus vllm#51820 (mooncake token_len
# clamp, needed by any arm with a mooncake tier under mtp) and the NIXL
# memory-region cap knob (default off, so it costs nothing unless an arm sets
# VLLM_NIXL_MAX_MR_BYTES).
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mcclamp,mrcap"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
