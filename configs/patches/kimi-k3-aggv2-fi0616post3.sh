#!/usr/bin/env bash
# AGG v2 runtime setup, on the FlashInfer version upstream now pins.
#
# kimi-k3-aggv2.sh force-installs 0.6.16rc5. vllm#50892 (2026-08-09) moved main to
# 0.6.16.post3, three post-releases later. This exists only so that bump can be
# priced on its own: it is the same script with one variable changed, so an arm
# using it differs from the rc5 arm in FlashInfer and nothing else.
#
# Do not fold this into the base script. The rc5 pin is deliberate -- it is what
# every AGG v2 number on the report was measured with -- and it stays until this
# A/B says otherwise.
set -euo pipefail

export FI_VER=0.6.16.post3
exec bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-aggv2.sh"
