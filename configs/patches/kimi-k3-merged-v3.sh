#!/usr/bin/env bash
# Runtime setup for k3-merged-v3: the same seven commits on today's upstream.
#
# v3 is v2's branch rebased from upstream 75231eff (2026-08-08) onto 0e2d78028c
# (2026-08-11), 49 commits later. It exists to answer one question -- what does
# moving to current main cost or buy -- so nothing else about it is new.
#
# FLASHINFER IS NOT HELD BACK. v2 force-installs 0.6.16rc5; #50892 moved main to
# 0.6.16.post3 and the v3 image is built against that. Downgrading here would
# measure "new upstream with old FlashInfer", which is not a configuration anyone
# would ship. It is pinned explicitly rather than left to the image so the
# version is in the log next to the numbers.
#
# WHY THE EXTRA MARKER. kimi-k3-merged-v2.sh checks four markers that v2 and v3
# both carry, so on its own it cannot tell the two images apart -- mispin v2 here
# and the run would succeed and be scored as a v3 result, which is the whole
# comparison. vllm/v1/worker/mamba_utils.py arrives with #49436 and does not
# exist in v2, so it separates them.
set -euo pipefail

export FI_VER=0.6.16.post3
bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-merged-v2.sh"

python3 - <<'PY'
import pathlib

import vllm

root = pathlib.Path(vllm.__file__).parent

# Reject a k3-merged-v2 image. Two accepted images, because the aarch64 build
# for GB300 is source-built from misunp/k3-wei-v2 rather than being the v3
# image plus runtime patches:
#
#   v3        vllm#49436's _memcpy_u64_tiled in v1/worker/mamba_utils.py.
#             Upstream has since renamed that helper to batch_memcpy, so the
#             symbol identifies the v3 image specifically and is absent from
#             anything newer -- which is how it failed every GB300 job.
#   k3-wei-v2 fd3e230e7's get_replay_boundary on the coordinator. Present in
#             neither v2 nor v3, so it cannot let a v2 image through.
CANDIDATES = [
    ("v1/worker/mamba_utils.py", "_memcpy_u64_tiled", "k3-merged-v3"),
    ("v1/core/kv_cache_coordinator.py", "def get_replay_boundary", "k3-wei-v2"),
]
found = [
    who for rel, mark, who in CANDIDATES
    if (root / rel).exists() and mark in (root / rel).read_text()
]
assert found, (
    "wrong image: none of "
    + "; ".join(f"{mark!r} in {rel} ({who})" for rel, mark, who in CANDIDATES)
    + ". Both k3-merged-v3 and a source build of misunp/k3-wei-v2 are accepted; "
    "a k3-merged-v2 image has neither."
)
print(f"=== image identified as {'/'.join(found)} ===")
import flashinfer  # noqa: E402
print(f"=== v3 image verified; flashinfer {flashinfer.__version__} ===")
PY
