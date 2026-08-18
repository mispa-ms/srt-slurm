#!/usr/bin/env bash
# vllm-container-deps-k3-b200-dcp8.sh plus one line: return the loader's cached
# blocks to the driver before the draft model is built.
# =============================================================================
# WHAT THIS FIXES. Every B200 arm with DSpark ns=7 and DCP8 + direct died the
# same way, and it was never DSpark and never symmetric memory:
#
#   dspark_mla.py:66  self.self_attn = MultiHeadLatentAttention(
#   mla.py:331        self.dcp_manager = MLADCPManager(
#   dcp_utils.py:645  direct_workspace = get_direct_dcp_a2a_workspace(
#   dcp_utils.py:137  storage = symm_mem.empty(shape, device=..., dtype=dtype)
#   RuntimeError: CUDA driver error: out of memory
#
# The draft model builds its own MLA, that MLA builds its own DCP manager, and
# that manager asks for a second direct-DCP A2A workspace. The target model's
# workspace was allocated during model construction, before any weights were
# read, so it succeeded. The draft's is allocated after, and by then:
#
#   INFO default_loader.py:430  Loading weights took 177.05 seconds
#   CUDACachingAllocator OOM ... (free: 1703936, total: 191503007744)
#
# 1.6 MB free of 178.35 GiB. The weights are not what filled it -- profiling on
# a surviving arm reports "Actual usage is 131.57 GiB for consumed memory
# (weights + non-torch)", so 46 GiB is genuinely unused. It is fastsafetensors'
# staging buffers, freed into the torch caching allocator and never returned to
# the driver. torch does not care: it reuses cached blocks, which is why the
# OOM lines above are warnings and the no-spec arms run to completion. But
# symm_mem.empty goes to the driver for physical memory, finds none, and dies.
#
# WHY gpu-memory-utilization CANNOT FIX THIS. gmu decides how much to give the
# KV cache after profiling; this failure happens before profiling, while the
# model is still loading, and nothing has read gmu yet. It also does not touch
# the caching allocator's reserve. The gmu 0.88 arm failed identically -- as did
# the mns 8, cap 4k, NCCL_CUMEM_ENABLE and B300-env arms, for the same reason.
#
# WHY B300 DOES NOT HIT IT. Same 131.57 GiB of weights against 288 GiB instead
# of 180. The loader's cache never reaches the ceiling there, so the driver
# still has room when the draft asks.
#
# So: one empty_cache() immediately before the speculator loads. It costs a
# handful of milliseconds once at startup, and gives the draft's workspace the
# 46 GiB that was already free in every sense except the driver's.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8.sh

echo "=== k3-b200-dcp8-emptycache: release cached blocks before the draft model ==="

python3 - <<'PY'
import importlib.util, os, sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/v1/worker/gpu/model_runner.py",
)
src = open(target).read()
if "[emptycache]" in src:
    print("[emptycache] already applied: " + target)
    sys.exit(0)

anchor = "                self.speculator.load_model(self.model)"
if src.count(anchor) != 1:
    sys.exit(
        "[emptycache] FATAL: anchor found %d times in %s" % (src.count(anchor), target)
    )

addition = """                # [emptycache] The draft model's MLA allocates its own direct-DCP
                # symmetric-memory workspace, which comes from the driver rather
                # than the caching allocator. After the target weights load, the
                # loader's freed staging buffers leave the driver with nothing.
                free_before, total = torch.cuda.mem_get_info()
                torch.cuda.empty_cache()
                free_after, _ = torch.cuda.mem_get_info()
                logger.info(
                    "[emptycache] driver-visible free before draft load: "
                    "%.2f -> %.2f GiB of %.2f GiB",
                    free_before / 2**30, free_after / 2**30, total / 2**30,
                )
""" + anchor

patched = src.replace(anchor, addition)
compile(patched, target, "exec")
open(target, "w").write(patched)
print("[emptycache] applied: " + target)
PY

echo "=== k3-b200-dcp8-emptycache: done ==="
