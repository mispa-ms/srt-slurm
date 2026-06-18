#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Patch LMCache's vllm_v1_adapter.request_finished for save_only_first_rank.
#
# Bug (lmcache 0.4.5..0.4.7 + dev): the FINISHED_ABORTED cleanup branch does
#   `assert self.lmcache_engine is not None`
# but with `save_only_first_rank: true` only the first rank holds an engine — every
# other rank has lmcache_engine=None. When the agentic workload ABORTS a request,
# request_finished runs on ALL ranks → the assert fires on non-first ranks →
# AssertionError → EngineCore dies (observed AIB #55199425 @ line 1849).
# (The rest of request_finished already guards `if self.lmcache_engine is not None`;
#  only the abort branch is missing the guard. This is what the turbo "lmcachefix"
#  container patches.)
#
# Fix: insert an early-return guard right before that assert so a no-engine rank
# returns cleanly during abort. Idempotent; anchors on the FINISHED_ABORTED block.

import importlib.util
import sys

MARKER = "save_only_first_rank guard (patched)"
ABORT_ANCHOR = "if request.status == RequestStatus.FINISHED_ABORTED:"
ASSERT = "assert self.lmcache_engine is not None"


def main() -> int:
    spec = importlib.util.find_spec("lmcache.integration.vllm.vllm_v1_adapter")
    if spec is None or not spec.origin:
        print("[lmcache-patch] WARN: lmcache vllm_v1_adapter not importable; skipping")
        return 0
    path = spec.origin
    src = open(path).read()

    if MARKER in src:
        print(f"[lmcache-patch] already applied ({path})")
        return 0

    a = src.find(ABORT_ANCHOR)
    if a == -1:
        print(f"[lmcache-patch] WARN: FINISHED_ABORTED anchor not found in {path}; skipping")
        return 0
    asrt = src.find(ASSERT, a)
    if asrt == -1:
        print("[lmcache-patch] WARN: assert not found after FINISHED_ABORTED; skipping")
        return 0

    line_start = src.rfind("\n", 0, asrt) + 1
    indent = src[line_start:asrt]  # leading whitespace of the assert line
    guard = (
        f"{indent}if self.lmcache_engine is None:  # {MARKER}\n"
        f"{indent}    return False, None\n"
        f"{indent}"
    )
    src = src[:line_start] + guard + src[asrt:]
    open(path, "w").write(src)
    print(f"[lmcache-patch] applied early-return guard before abort assert ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
