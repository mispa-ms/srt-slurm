#!/usr/bin/env bash
# Let TokenSpeed MLA run MTP with a non-trivial CP interleave size.
# =============================================================================
# WITHOUT THIS, EVERY SPECULATIVE ARM DIES BEFORE SERVING on nightlies that carry
# vllm#50611. The chain is three steps and none of them is about disaggregation,
# even though the PR that introduced it is:
#
#   1. vllm/config/vllm.py, adjust_dcp_kv_cache_interleave_size(): when DCP > 1 and
#      *any* kv_connector is configured, cp_kv_cache_interleave_size is promoted
#      from 1 to the local block_size. Our AGG arms set MooncakeStoreConnector, so
#      this fires on aggregated serving too -- the guard only asks whether a
#      connector exists, not whether we are doing P/D.
#   2. vllm/v1/worker/cp_utils.py:41: with a speculative_config and interleave > 1,
#      it asserts layer_impl.supports_mtp_with_cp_non_trivial_interleave_size.
#   3. TokenSpeedMLAImpl does not set that attribute, so it inherits False from
#      vllm/v1/attention/backend.py:812 and the assert raises:
#
#        AssertionError: MTP with cp_kv_cache_interleave_size > 1 is not
#        supported in TokenSpeedMLAImpl.
#
# The 2026-08-28 image (6f7df92a) has the assert but NOT the promotion --
# adjust_dcp_kv_cache_interleave_size does not exist there -- so interleave stays 1
# and the assert never fires. The promotion arrived with
# 7f4793eaa3 "[Nixl][PD] DCP support for MLA models (#50611)". That is what makes
# this a new wall between 08/28 and 09/01 for our exact recipe: DCP8 +
# MooncakeStoreConnector + DSpark + attention-backend TOKENSPEED_MLA.
#
# The fix is Wei Zhao's e10c92e116 on wzhao18/vllm @ wzhao/k3-nvfp4-perf, one line
# declaring the capability. vllm#54457 would fix it from the other side by adding
# requires_dcp_block_aligned_interleave so the promotion stops over-firing, but it
# is not merged and it misses MooncakeStoreConnector anyway.
#
# This is a capability declaration, not a behaviour change: it says the backend
# tolerates the interleave, which the FlashInfer MLA backend already declares.
# =============================================================================
set -euo pipefail

echo "=== tokenspeed-mtp-interleave: allow MTP with cp interleave > 1 ==="

VLLM_ROOT=$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
target = os.path.join(root, "vllm/v1/attention/backends/mla/tokenspeed_mla.py")
if not os.path.exists(target):
    sys.exit("[tokenspeed-mtp] FATAL: no tokenspeed_mla.py in this image")

src = open(target).read()
FLAG = "supports_mtp_with_cp_non_trivial_interleave_size"

if FLAG in src:
    print("[tokenspeed-mtp] already present in this image")
    sys.exit(0)

# Anchor on the sibling capability the backend already declares, so the new line
# lands beside the others rather than at an arbitrary point in the class body.
ANCHOR = "    supports_non_causal_multi_token_dcp: ClassVar[bool] = True\n"
if src.count(ANCHOR) != 1:
    sys.exit(
        "[tokenspeed-mtp] FATAL: expected one supports_non_causal_multi_token_dcp "
        "declaration, found %d" % src.count(ANCHOR)
    )

src = src.replace(ANCHOR, ANCHOR + "    " + FLAG + ": ClassVar[bool] = True\n", 1)
compile(src, target, "exec")
open(target, "w").write(src)
print("[tokenspeed-mtp] applied: " + target)
PY

# Verify against the assertion that would fire, not just against the file we wrote:
# read the flag off the class the way cp_utils.py reads it off the instance.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/v1/attention/backends/mla/tokenspeed_mla.py")).read()
if "supports_mtp_with_cp_non_trivial_interleave_size" not in src:
    sys.exit("[tokenspeed-mtp] FATAL: the flag is not in the file after writing it")

cp = os.path.join(root, "vllm/v1/worker/cp_utils.py")
if os.path.exists(cp):
    guard = open(cp).read()
    if "supports_mtp_with_cp_non_trivial_interleave_size" not in guard:
        print("[tokenspeed-mtp] note: this image has no MTP interleave assert to satisfy")
    else:
        print("[tokenspeed-mtp] verified: the assert in cp_utils.py is now satisfiable")
else:
    print("[tokenspeed-mtp] note: no cp_utils.py in this image")
PY

echo "=== tokenspeed-mtp-interleave: done ==="
