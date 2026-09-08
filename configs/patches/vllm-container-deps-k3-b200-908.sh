#!/usr/bin/env bash
# Kimi-K3 B200 on the 2026-09-08 nightly (9ea8f3ff).
# =============================================================================
# This chain is almost empty, and that is the news. Everything we carried into
# the 09/01 attempt except int64idx is now upstream, verified against this tree:
#
#   #50514  [Core][MRV2] Support eagle3 spec decode with pipeline parallel,
#           merged as d87a440f88. It covers dspark, not just eagle3. The refusal
#           our carry deleted ("<method> with pipeline parallel is not supported")
#           is gone from model_runner.py and supports_aux_hidden_states_over_pp is
#           present, so the 596-line k3-dspark-pp carry is retired rather than
#           re-targeted -- 27 of its hunks no longer apply because the code they
#           add is already there.
#   #50611's over-promotion is fixed. adjust_dcp_kv_cache_interleave_size now
#           returns early unless the transfer config has a NixlConnector, so it
#           no longer fires for MooncakeStoreConnector on plain AGG. Our
#           interleave narrowing is unnecessary, and so is the TokenSpeed MTP
#           capability flag that only mattered while interleave was promoted.
#   The HF snapshot problem is gone with it: upstream's dspark/utils.py does not
#           resolve the model directory through snapshot_download at all, so the
#           IncompleteSnapshotError that killed every speculative arm on 09/01 --
#           13 missing .eval_results/*.yaml in the staged checkpoint -- cannot
#           happen here.
#   #53324  absorbed (connector.py reads `transfer_groups > 1 and pcp > 1`).
#   #54167  absorbed (the mixin chains to super().__init__()).
#
# STILL OURS: int64idx. kda.py still loads a 32-bit state_idx and multiplies it
# by state_stride_0, and without the cast the run dies around 19.5 minutes.
#
# NOT INCLUDED: mambagroups2 / mambacache. They serve `mamba-cache-mode align`,
# which these arms do not set. mambagroups2 in particular has a FATAL gate that
# killed all five arms of pipeline 66109112 on the 09/01 image.
#
# So a difference against the 08/28 numbers is ten days of upstream plus
# upstream's own spec-decode-under-PP implementation in place of our carry --
# which is itself worth measuring, since the two are not the same code.
# =============================================================================
set -euo pipefail

echo "=== k3-b200-908: chain for the 2026-09-08 nightly ==="

bash /configs/patches/vllm-container-deps-k3-b200-828.sh
bash /configs/patches/vllm-container-deps-k3-b200-dcp8-emptycache.sh
# NOT in the original 908 chain, and the reason all four 09/08 arms died: upstream
# #50514 builds the drafter on the last PP stage only, but the draft loader inherits
# load_format=fastsafetensors, whose iterator collectives over group.WORLD. PP0
# never joins and PP1 waits out the NCCL timeout on a 1-element broadcast. Our
# retired carry had this guard; #50514 does not.
bash /configs/patches/vllm-container-deps-k3-dspark-draft-loader.sh

# Confirm the two things this chain is betting on, rather than assuming them: that
# int64idx really landed, and that spec decode under PP is present without our
# carry. If #50514 were absent the speculative arms would refuse to start and the
# failure would look like a config problem instead of a missing upstream commit.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
kda = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda.py")).read()
runner = open(os.path.join(root, "vllm/v1/worker/gpu/model_runner.py")).read()
iface = open(os.path.join(root, "vllm/model_executor/models/interfaces.py")).read()

if "checkpoint_state_indices_ptr + seq_idx).to(tl.int64)" not in kda:
    sys.exit("[k3-b200-908] FATAL: the int64 state_idx cast is not present")
if "with pipeline parallel is not supported" in runner:
    sys.exit(
        "[k3-b200-908] FATAL: this image still refuses spec decode under PP; "
        "#50514 is not in it and the dspark-pp carry would be needed"
    )
if "supports_aux_hidden_states_over_pp" not in iface:
    sys.exit("[k3-b200-908] FATAL: supports_aux_hidden_states_over_pp is missing")
print("[k3-b200-908] verified: int64 cast present, spec-under-PP available upstream")
PY

echo "=== k3-b200-908: done ==="
