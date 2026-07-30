#!/bin/bash
# Apply the standard Kimi-K3 setup and the two patches needed to exercise
# DS-layout DSpark with NIXL Mamba prefix-cache state transfer.

set -euo pipefail

bash /configs/patches/vllm-container-deps-remove-ds-layout-spec-assert.sh

target=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
patch_file=/configs/patches/vllm-nixl-ssm-spec-prefix-cache.patch
old_assertion="SSM can only have one local block"
patched_marker="SSM expected {num_state_slots} local state slots"

if grep -Fq "${old_assertion}" "${target}"; then
    patch --batch --forward -p1 \
        -d /usr/local/lib/python3.12/dist-packages < "${patch_file}"
elif grep -Fq "${patched_marker}" "${target}"; then
    echo "NIXL SSM speculative prefix-cache patch is already applied"
else
    echo "ERROR: unexpected NIXL base_worker.py; refusing to patch" >&2
    exit 1
fi

if grep -Fq "${old_assertion}" "${target}" ||
   ! grep -Fq "${patched_marker}" "${target}"; then
    echo "ERROR: NIXL SSM speculative prefix-cache patch verification failed" >&2
    exit 1
fi

