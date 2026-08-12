#!/usr/bin/env bash
# Time the MLA gather kernels on this image, before the server starts.
#
# The revert A/B answers "does the regression go away". It cannot answer "was it
# this kernel", because a full serving run moves everything at once. This does:
# the same script, the same shapes, run against the kernel from before #51739 and
# the kernel from after, on the same GPU.
#
# Sourced by the arms that carry it rather than by kimi-k3-merged-v4-mooncake.sh,
# so the ladder arms are not slowed by it and their logs are not changed.
#
# Costs seconds and runs on the allocated node before the model loads, so it adds
# no GPU time that the serving run was not already going to use.
#
# maybe-dequant is our branch (kv-cache-dtype fp8). cache and fp8-upconvert are
# the other two the same commit rewrote; they are timed too because if all three
# move together the cause is the rewrite, and if only one moves it is narrower
# than that.
set -euo pipefail

BENCH=/configs/patches/k3_gather_microbench.py

echo "=== MLA gather microbenchmark (vllm#51739 A/B) ==="
python3 -c "import vllm, torch; print(f'vllm {vllm.__version__}  torch {torch.__version__}  {torch.cuda.get_device_name(0)}')"

# skew-8 is the shape this workload actually presents -- eight sequences of
# uneven length sharing one chunked-context gather -- and single-300k is the
# stress case. Both are upstream's own scenarios, unmodified.
for scen in single-60k skew-8 single-300k; do
    echo "--- scenario ${scen}"
    python3 "$BENCH" --variant all --scenario "$scen" || {
        echo "microbenchmark failed for ${scen}; continuing, the serving run is the deliverable"
        continue
    }
done
echo "=== end MLA gather microbenchmark ==="
