#!/usr/bin/env bash
# The 2026-09-08 chain with int64idx deliberately left out.
# =============================================================================
# This arm exists to fail, and to record exactly how.
#
# int64idx is the last patch we still carry. Everything else we needed on 09/01
# is upstream in this image (#50514, #50611's narrowing, #53324, #54167), so if
# this one is real it deserves an upstream PR rather than a private carry -- and
# a PR needs a reproduction on a current nightly, not a claim about an August
# image.
#
# WHAT WE EXPECT. vllm/models/kimi_k3/nvidia/kda.py loads state_idx as a 32-bit
# value in _store_cache_checkpoints_kernel and then multiplies it by
# state_stride_0. On K3 that stride is 442,368, so the product overflows int32
# once state_idx passes ~4,854 and the address wraps. Serving comes up fine and
# dies around 19.5 minutes in, when enough checkpoints have been written for the
# index to get that far. So a clean start here is not evidence of anything --
# the run has to be read at its end.
#
# The argument the PR will make is that main already applies the identical fix
# for the identical reason in vllm/v1/worker/mamba_utils.py; only the K3 kernel
# was missed.
#
# Its twin, vllm-container-deps-k3-b200-908.sh, is this chain plus the cast, and
# the two differ in nothing else.
# =============================================================================
set -euo pipefail

echo "=== k3-b200-908-noint64: the 09/08 chain WITHOUT the int64 cast ==="

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-pr54167.sh
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh

# Assert the omission, so a silently-already-fixed image cannot be mistaken for a
# reproduction. If upstream lands the cast, this arm stops being meaningful and
# should say so loudly rather than pass.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
kda = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()

if "checkpoint_state_indices_ptr + seq_idx).to(tl.int64)" in kda:
    sys.exit(
        "[908-noint64] FATAL: this image already casts state_idx to int64 -- "
        "the carry is upstream and this arm has nothing to reproduce"
    )
if "state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx)" not in kda:
    sys.exit("[908-noint64] FATAL: the state_idx load is not where we expect it")
if "state_idx * state_stride_0" not in kda:
    sys.exit("[908-noint64] FATAL: the conv_state address no longer uses state_stride_0")
print("[908-noint64] confirmed: 32-bit state_idx multiplied by state_stride_0, uncast")
PY

echo "=== k3-b200-908-noint64: done ==="
