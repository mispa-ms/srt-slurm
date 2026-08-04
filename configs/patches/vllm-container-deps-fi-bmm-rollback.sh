#!/usr/bin/env bash
# Point FlashInfer's trtllm-gen BatchedGemm cubin bundle back to the one
# FlashInfer 0.6.11.post2 pins, leaving everything else on the installed version.
# =============================================================================
# WHY: the expert BatchedGemm kernels are not compiled by FlashInfer; they are
#   prebuilt cubins fetched from artifactory, and flashinfer/artifacts.py only
#   holds the bundle path plus its checksum. 0.6.11.post2 and 0.6.15.post1 pin
#   different bundles:
#
#     0.6.11.post2  3d9dd08b.../batched_gemm-4fc8a68-6743435/
#     0.6.15.post1  b368d003.../batched_gemm-da58956-b4ac80e/
#
#   On B300 NVFP4 Kimi-K2.5 the newer bundle runs the expert BatchedGemm ~3x
#   slower at c16 with identical grid/block/registers, and disabling the
#   autotuner does not recover it -- consistent with a different candidate set
#   rather than a worse ranking. Rolling back only this string isolates the
#   cubin bundle from every other change between the two FlashInfer versions.
#
# Only TRTLLM_GEN_BMM is touched. FMHA and GEMM keep the installed bundles.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps.sh

python3 - <<'PY'
import pathlib
import re

import flashinfer.artifacts as A

OLD_PATH = "3d9dd08b1691e63e298a7b862d74fd7af3daf594/batched_gemm-4fc8a68-6743435/"
OLD_SUM = "44174e2a08bb427088f5b5443bf0108bb6fb6cb0812ff6018f6418b3d2273824"

src = pathlib.Path(A.__file__)
text = src.read_text()
before = (A.ArtifactPath.TRTLLM_GEN_BMM, A.CheckSumHash.TRTLLM_GEN_BMM)
print(f"[fi-bmm] installed bundle   {before[0]}")
print(f"[fi-bmm] installed checksum {before[1]}")

if before[0] == OLD_PATH:
    raise SystemExit("[fi-bmm] FATAL: already on the target bundle; this arm would be a no-op")

# Replace the two literals in place. Both are unique in the file, so an exact
# count check is enough to catch an upstream rename.
for old, new, label in ((before[0], OLD_PATH, "path"), (before[1], OLD_SUM, "checksum")):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[fi-bmm] FATAL: expected 1 occurrence of the {label}, found {n}")
    text = text.replace(old, new)

src.write_text(text)
PY

# Re-import in a fresh interpreter so we read what actually landed on disk.
python3 - <<'PY'
import flashinfer.artifacts as A

print(f"[fi-bmm] now bundle   {A.ArtifactPath.TRTLLM_GEN_BMM}")
print(f"[fi-bmm] now checksum {A.CheckSumHash.TRTLLM_GEN_BMM}")
assert A.ArtifactPath.TRTLLM_GEN_BMM.endswith("batched_gemm-4fc8a68-6743435/"), "rollback did not stick"
PY

echo "=== fi-bmm-rollback: done ==="
