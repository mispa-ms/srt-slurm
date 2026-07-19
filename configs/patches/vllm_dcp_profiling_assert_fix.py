# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Patch vLLM's prepare_dcp_dummy_context_metadata to tolerate the minimal KV
cache config used during memory profiling.

Symptom (seen on GB300, DISAGG, decode-context-parallel-size=4):
    File ".../vllm/v1/worker/cp_utils.py", in prepare_dcp_dummy_context_metadata
        assert max_valid_block_id > 0
    AssertionError
during determine_available_memory -> profile_cudagraph_memory (decode worker
crashes on cudagraph capture).

Root cause: PR #40996 ("DCP supports hybrid attention", commit 95ed0feaa5)
added prepare_dcp_dummy_context_metadata, which populates dummy KV block-table
entries for DCP graph warmup and asserts kv_cache_config.num_blocks >= 2. But
during determine_available_memory, gpu_model_runner._init_minimal_kv_cache_for_
profiling installs a *minimal placeholder* KV config (num_blocks <= 1); the
real KV sizing happens afterwards. On a disagg decode-only worker the profiling
cudagraph is uniform_decode, so get_dcp_dummy_context_len returns nonzero and
the assert fires on the placeholder. (AGG workers are mixed prefill+decode, so
get_dcp_dummy_context_len returns 0 there and never reaches the assert -- which
is why AGG DCP4 works and disagg DCP4 crashes.)

Fix: mirror the existing `if dcp_dummy_context_len == 0: return` early-out.
When the KV config is the minimal profiling placeholder (num_blocks <= 1) there
are no real blocks to point at, so skip the dummy-context fill entirely. This
matches pre-#40996 behavior; the real cudagraph capture after KV sizing has
many blocks and is unaffected. NB: simply removing the assert is wrong --
`block_idx % max_valid_block_id` (below) would then ZeroDivisionError.

Reference: vllm/v1/worker/cp_utils.py, prepare_dcp_dummy_context_metadata().
Unfixed in vLLM main as of 2026-07-19 (ace9fda495).
"""

import importlib.util
import sys
from pathlib import Path


def _find_cp_utils() -> Path | None:
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            candidate = Path(root) / "v1" / "worker" / "cp_utils.py"
            if candidate.exists():
                return candidate
    # Fallback: common dist-packages locations across python minor versions.
    for py in ("python3.12", "python3.11", "python3.10"):
        candidate = Path(
            f"/usr/local/lib/{py}/dist-packages/vllm/v1/worker/cp_utils.py"
        )
        if candidate.exists():
            return candidate
    return None


# Idempotency marker: unique to our inserted guard.
MARKER = "if kv_cache_config.num_blocks <= 1:"

# Anchor: the assert immediately after the num_blocks read.
OLD = (
    "    max_valid_block_id = kv_cache_config.num_blocks - 1\n"
    "    assert max_valid_block_id > 0\n"
)

NEW = (
    "    # srt-slurm-sa hotfix (PR #40996): during determine_available_memory\n"
    "    # profiling the KV config is a minimal placeholder (num_blocks <= 1);\n"
    "    # there are no real blocks to point at, so skip the dummy-context fill\n"
    "    # (matches pre-#40996 behavior). Real capture after KV sizing is fine.\n"
    "    if kv_cache_config.num_blocks <= 1:\n"
    "        return\n"
    "    max_valid_block_id = kv_cache_config.num_blocks - 1\n"
)


def main():
    target = _find_cp_utils()
    if target is None:
        print(
            "[vllm-dcp-profiling-assert-fix] cp_utils.py not found; "
            "vLLM layout may have changed. Skipping (non-fatal).",
            file=sys.stderr,
        )
        # Non-fatal: older images (pre-#40996) do not have this file/path and
        # do not need the patch.
        return

    content = target.read_text()

    if MARKER in content:
        print(
            "[vllm-dcp-profiling-assert-fix] Already patched, skipping.",
            file=sys.stderr,
        )
        return

    count = content.count(OLD)
    if count == 0:
        print(
            "[vllm-dcp-profiling-assert-fix] Anchor not found in "
            f"{target}; likely a pre-#40996 image (no patch needed) or a "
            "drifted version. Skipping (non-fatal).",
            file=sys.stderr,
        )
        return
    if count > 1:
        print(
            f"[vllm-dcp-profiling-assert-fix] Anchor is ambiguous ({count} "
            "occurrences); refusing to patch.",
            file=sys.stderr,
        )
        sys.exit(1)

    content = content.replace(OLD, NEW)
    target.write_text(content)
    print(
        f"[vllm-dcp-profiling-assert-fix] Patched {target}: skip dcp dummy "
        "context fill when num_blocks <= 1 (profiling placeholder).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
