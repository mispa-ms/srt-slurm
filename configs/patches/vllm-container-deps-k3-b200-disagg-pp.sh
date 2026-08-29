#!/usr/bin/env bash
# B200 disaggregated serving with a pipeline-parallel prefill and decode.
# =============================================================================
# The AGG chain plus the one thing PP disagg needs on top of it.
#
# WHY BOTH HALVES ARE PP2 ON B200, unlike GB300 where the decode side is a single
# TP8 worker: K3 MXFP4 is ~1.3 TiB of weights, so a TP8 worker holds ~163 GiB before
# non-torch. That fits GB300's 288 GiB and does not fit B200's 178. Sixteen chips is
# the floor for a K3 worker here, and TP16 is not the way to reach it -- the NIXL
# handshake compares a Mamba block ratio across the pair and TP moves that ratio while
# PP does not. So both sides are TP8 x PP2, which also keeps the ratio identical.
#
# WHAT THE SECOND SCRIPT IS FOR. With PP > 1 the store connector refuses PP-sharded
# handshake metadata and engine-core init dies on every rank past the first stage.
# MooncakeStoreConnector never reads that metadata -- producer and consumer meet in
# MooncakeDistributedStore -- so the refusal kills startup over a value it discards.
# vllm-container-deps-k3-mooncake-pp-handshake.sh is the no-op override, and it is
# written against the images this workstream runs (connector.py:172 on 728d3ad).
#
# DO NOT ADD UCX_TLS TO THE CONFIG THAT USES THIS. The GB300 disagg configs set
# UCX_TLS=rc,cuda_copy; an explicit allowlist is the documented cause of NIXL
# "failed to set active message handler" and UCX mpool exhaustion, confirmed across
# several teams. Let UCX auto-detect.
# =============================================================================
set -euo pipefail

echo "=== k3-b200-disagg-pp: AGG chain + the PP handshake override ==="

# THE CHECKPOINT PATH HAS TO BE SET HERE, NOT IN THE CONFIG. vllm-container-deps-k3-hfshim.sh
# resolves ${K3_STAGED_DIR:-/lustre/share/coreai_comparch_aarwlt/...}, the bia path, and
# refuses to continue rather than let HF start a 1.4 TB download. Our AGG configs set the
# variable in `aggregated_environment` and it arrives; a disagg config setting it in
# `prefill_environment` / `decode_environment` does NOT reach this preamble -- two
# submissions died on the bia default with the variable present in the YAML. Rather than
# keep guessing which block propagates, default it here and let a config still override.
export K3_STAGED_DIR="${K3_STAGED_DIR:-/lustre/share/coreai_comparch_inferencex/models/kimi-k3}"
echo "[k3-b200-disagg-pp] K3_STAGED_DIR=${K3_STAGED_DIR}"

bash /configs/patches/vllm-container-deps-k3-b200-dspark-pp.sh
bash /configs/patches/vllm-container-deps-k3-mooncake-pp-handshake.sh

python3 - <<'PY'
import importlib.util, os, sys
root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
conn = os.path.join(root, "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py")
src = open(conn).read()
if "set_xfer_handshake_metadata_pp_aware" not in src:
    sys.exit("[k3-b200-disagg-pp] FATAL: the PP handshake override is not in the store connector")
print("[k3-b200-disagg-pp] verified: store connector accepts PP-sharded handshake metadata")
PY

echo "=== k3-b200-disagg-pp: done ==="
