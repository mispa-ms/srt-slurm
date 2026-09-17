#!/usr/bin/env bash
# 0915a plus mcppdbg4 only: the store scheduler logs its load decision per
# request. Standalone -- the worker-side debug patches are deliberately absent.
# See kimi-k3-gb300-pp-911.sh.
export K3_OURS="ssm,mcclamp,mcpp,mrcap,pushdone,mcppdbg4"
exec bash /configs/patches/kimi-k3-gb300-pp-911.sh
