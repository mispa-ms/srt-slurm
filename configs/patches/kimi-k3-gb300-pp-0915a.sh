#!/usr/bin/env bash
# 09-15 nightly (cd10ed6f), PP arm: K3_OURS=ssm,mcclamp,mcpp,mrcap,pushdone.
#
# Same as 0914b plus `pushdone`, which undoes vllm#56104's `_recving_metadata`
# gate on the push producer's send completions. #56104 landed between the 09-14
# and 09-15 nightlies; without the undo a prefill worker never reports a push
# finished and every request waits out its 30 s lease.
#
# The step self-skips on an image that predates #56104, so this wrapper is also
# safe on 09-14.
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mcclamp,mcpp,mrcap,pushdone"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
