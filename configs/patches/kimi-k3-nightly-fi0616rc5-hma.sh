#!/bin/bash
# kimi-k3-nightly-fi0616rc5.sh with the hybrid-KV recompute patch asserted.
#
# The base script applies patch_kimi_k3_mooncake_hma_recompute.py behind a
# `|| echo NOTE`, so a missing file reads as a note rather than a failure.
# Pipeline 61123650 lost four B300 jobs to exactly that: the file was absent from
# this branch, the note scrolled past, and the engine died eleven minutes into the
# benchmark with `ValueError: too many values to unpack (expected 1)` at
# scheduler.py:_update_requests_with_invalid_blocks. Mooncake makes external KV
# loads fail occasionally, which is what reaches that path; the SimpleCPUOffload
# arms never did. Assert it rather than trusting the note.
#
# This is the -mlaws.sh script minus the MLA chunked-prefill workspace patch, and
# -mlaws.sh calls this file, so the two differ by exactly that one patch. An arm
# on this script and an arm on -mlaws.sh with everything else held fixed prices
# wzhao18/vllm@2331dddd94 + @d4f1b6438c on its own.
#
# Expect this script to be the one that OOMs when max-num-seqs is left unset:
# without the workspace patch, K3's block_size of 1536 puts the floor at
# 1024 * 1536 tokens, ~10.5 GB per GPU at TP8 against 420 MB at the 64k cap.
# A failure to start here is a result, not a broken run.

set -euo pipefail

bash /configs/patches/kimi-k3-nightly-fi0616rc5.sh

echo "=== hybrid-KV recompute patch ==="
python3 /configs/patches/patch_kimi_k3_mooncake_hma_recompute.py
python3 -c "
import vllm, pathlib
p = pathlib.Path(vllm.__file__).parent / 'v1/core/sched/scheduler.py'
src = p.read_text()
assert 'req_hybrid_block_ids = {' in src, 'hybrid-KV recompute patch not present'
assert '(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)' not in src, \
    'the single-group unpack that crashes hybrid models is still there'
print('hybrid-KV recompute patch verified in', p)
"
