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
rel, mark = "v1/worker/mamba_utils.py", "_memcpy_u64_tiled"
path = root / rel
assert path.exists() and mark in path.read_text(), (
    f"wrong image: expected k3-merged-v3, but {rel} is missing or does not "
    f"contain {mark!r}. That file arrives with vllm#49436 and is absent from "
    "k3-merged-v2, so this is a v2 image."
)
import flashinfer  # noqa: E402
print(f"=== v3 image verified; flashinfer {flashinfer.__version__} ===")
PY
