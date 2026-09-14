#!/bin/bash
# The pdx setup with healthy-run sampling on, and nothing else changed.
#
# WHY A SEPARATE SCRIPT. The config's `environment:` block is applied AFTER the
# setup script runs, so an env-var gate inside the main script would always read
# unset -- the same trap that cost four runs with HF_HOME. `setup_script:` is a
# config field, so a wrapper that exports and then calls the base script is the
# only way to turn one of its gates on from a config.
#
# WHAT THIS ARM IS FOR. Every stall dump shows the store's
# KVCacheStoreSendingThread inside cuEventDestroy_v2 -- 7 of 8 ranks in 427243,
# 6 of 8 in 427249, and the same ranks again in that run's second dump seconds
# later, so they are stuck there rather than sampled there. That is only
# evidence if the thread does NOT normally sit there, and nothing has ever
# sampled a healthy run: the watchdog arms, waits, and fires once.
#
# So this arm samples one worker every 30 engine ticks (~5 min) while the run is
# healthy, up to 6 times. It is the control for the `-evtpool` arm and changes
# nothing else, so it doubles as another draw of the stall itself.
set -euo pipefail

export PDX_HEALTHY_DUMP_EVERY=30
export PDX_HEALTHY_DUMP_MAX=6

echo "=== evtctl: healthy sampling every ${PDX_HEALTHY_DUMP_EVERY} ticks," \
     "max ${PDX_HEALTHY_DUMP_MAX} samples ==="
bash /configs/patches/kimi-k3-wei-prebuilt-pdx.sh
echo "=== evtctl: done ==="
