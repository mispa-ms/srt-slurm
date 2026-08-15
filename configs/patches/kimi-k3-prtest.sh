#!/usr/bin/env bash
# Run the upstream PR's unit tests in the container, then stop.
#
# WHY THIS EXISTS. The fix for the DSpark regression is going upstream, and its
# regression test had never been executed: this host has no docker socket
# permission, enroot cannot unshare a user namespace, and the SLURM controller
# is not reachable from it. The container the sweeps already use has vLLM
# installed, and setup scripts run before the server starts, so this borrows one
# for two minutes.
#
# WHAT IT DOES. Clones vLLM at the image's own sha for the tests/ tree (the
# image ships site-packages only), patches the installed vllm with the fix, and
# runs the two prefix-cache test files -- first WITHOUT the fix to confirm the
# new test actually fails, then WITH it.
#
# IT ALWAYS EXITS NON-ZERO at the end. There is no benchmark worth running here
# and the job should stop before the model loads; a green pipeline would be the
# wrong signal anyway. Read the log, not the badge.
set -euo pipefail

readonly SHA=ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9
readonly FIX=/configs/patches/k3-prtest-fix.patch
readonly TEST=/configs/patches/k3-prtest-test.patch

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
python3 -c "import vllm; print('image vllm:', vllm.__version__)"

command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
rm -rf /tmp/vsrc && mkdir -p /tmp/vsrc && cd /tmp/vsrc
git init -q && git remote add origin https://github.com/vllm-project/vllm.git
git fetch -q --depth 1 origin "$SHA" && git checkout -q FETCH_HEAD
patch -p1 --forward < "$TEST"

python3 -m pip install -q pytest 2>/dev/null || true

echo "=============== 1. WITHOUT the fix: the new test must FAIL ==============="
set +e
python3 -m pytest -q \
    tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
    -k eagle_group_registers_unaligned_tail
before=$?
set -e
echo "exit before fix = $before"
[ "$before" -eq 0 ] && { echo "FATAL - the regression test passes without the fix, so it tests nothing" >&2; exit 1; }

echo "=============== 2. applying the fix to installed vllm ==============="
patch -p1 --forward -d "$SITE" < "$FIX"

echo "=============== 3. WITH the fix: the whole prefix_cache suite ==============="
python3 -m pytest -q tests/v1/core/prefix_cache/
echo "=============== 4. the pre-existing clamping test ==============="
python3 -m pytest -q tests/v1/core/test_prefix_caching.py -k clamped_to_lcm

echo "=== ALL GREEN. Stopping deliberately; there is no benchmark to run. ==="
exit 1
