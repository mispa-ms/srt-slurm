#!/usr/bin/env bash
# Kimi-K3 GB300 PP + disagg on the 2026-09-11 nightly.
# =============================================================================
# BASE. vllm/vllm-openai:nightly-e7edf17cea217e52701f913cd8491fcacf2d9490
# (upstream e7edf17cea, 2026-09-11 04:53 UTC; arm64 + amd64 on the hub).
#
# PURPOSE. Same GB300 PP2 disagg shape as the 09-08 and 09-09 runs, on the first
# nightly that carries vllm#55499 — the FlashInfer 0.6.18 per-layer D2H sync in
# TRTLLM ragged prefill, which every run from 08/30 to 09/09 was paying. Our
# prefill is TRTLLM_RAGGED, so that fix lands on the critical path here.
#
# Differences from kimi-k3-gb300-pp-909.sh, all because upstream absorbed work:
#  - #55745 (PP draft-broadcast record_stream) MERGED 09-09 -> step dropped.
#  - #55531 (symmetric DCP disagg for hybrid mamba, wzhao18) MERGED 09-10. It
#    removes the worker hybrid+DCP refusal outright, so our pushdcp patch is now
#    just the NixlPushConnector.__init__ raise plus the handshake equality check.
#  - #50499 rebased onto the renamed #50494 (head d1ac007938): the member
#    vocabulary became layer vocabulary (_member_local_regions ->
#    _transfer_layer_region_indices, _set_region_members -> _set_region_layers,
#    region_members -> region_layers). Both pr50499 and our ssm patch are recut.
#  - #55924 / #55472 remain merged; their steps take the no-op exits.
#  - #54416 (draft loader over group.WORLD) is still unmerged -> guard kept.
#
# CHAIN, EVERY ARM, IN ORDER: hfshim -> ckptidx-829 (no-op) ->
# dspark-draft-loader -> dspark-pr55472 (no-op) -> pr50499-911 -> pushdcp-911
# -> [ssm-911] -> [mcpp-908].
#
# OURS, BY K3_OURS:
#  ssm  -- SSM/Mamba layers over the layer-name path (necessary: the 09-08 A/B
#          arm A died at the Mamba-under-PP refusal, pipeline 66840854).
#  mcpp -- MooncakeStore PP handshake override (necessary with a mooncake tier:
#          0908bmc died at 'received pp_rank > 0 handshake metadata').
#
# Replayed on a pristine e7edf17ce tree with exit codes asserted per step.
# =============================================================================
set -euo pipefail
K3_OURS="${K3_OURS:-}"
echo "[k3-pp-911] K3_OURS='${K3_OURS}'"
if [ -z "${K3_STAGED_DIR:-}" ]; then
    for _cand in \
        /lustre/share/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/hanjieq/models/kimi-k3 \
        /lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3
    do
        if [ -d "${_cand}" ]; then
            export K3_STAGED_DIR="${_cand}"
            echo "[k3-pp] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3-pp] FATAL: no staged checkpoint on this cluster. Set K3_STAGED_DIR." >&2
    echo "[k3-pp] Refusing to continue -- the fallback is a 1.45 TB download inside" \
         "a 4-hour job, which fails later and less legibly than this does." >&2
    exit 1
fi

bash /configs/patches/vllm-container-deps-k3-hfshim.sh
bash /configs/patches/vllm-container-deps-k3-ckptidx-829.sh
bash /configs/patches/vllm-container-deps-k3-dspark-draft-loader.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pr55472.sh
bash /configs/patches/vllm-container-deps-k3-pr50499-911.sh
bash /configs/patches/vllm-container-deps-k3-pushdcp-911.sh
case ",${K3_OURS}," in *,ssm,*)  bash /configs/patches/vllm-container-deps-k3-ssm-911.sh ;; esac
case ",${K3_OURS}," in *,mcpp,*) bash /configs/patches/vllm-container-deps-k3-mcpp-908.sh ;; esac
K3_OURS="${K3_OURS}" python3 - <<'PY'
import importlib.util, os, sys
root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
ours = {x for x in os.environ.get("K3_OURS", "").split(",") if x}
def src(p): return open(os.path.join(root, p)).read()
fail = []
if "super().__init__(kv_cache_spec, layer_names, vllm_config, device)" not in src("vllm/v1/attention/backends/gdn_attn.py"):
    fail.append("not a >= 2026-09-05 tree: GDN builder still skips its base __init__")
if "aux_hidden_states_over_pp" not in src("vllm/models/kimi_k3/nvidia/model.py"):
    fail.append("#50514 (spec decode under PP) missing")
if "to(tl.int64)" not in src("vllm/models/kimi_k3/nvidia/kda.py"):
    fail.append("int64 checkpoint index missing (#55924)")
du = src("vllm/v1/worker/gpu/spec_decode/dspark/utils.py")
if 'load_config=replace(draft_vllm_config.load_config, load_format="auto")' not in du:
    fail.append("draft loader still uses fastsafetensors under PP (deadlock on the PP2 prefill)")
if "parallel_config=replace(vllm_config.parallel_config,pipeline_parallel_size=1" not in "".join(du.split()):
    fail.append("#55472 missing: the draft parallel config drops DCP (MLADCPManager assert on decode)")
if "idx_mapping.record_stream(self.broadcast_stream)" not in src("vllm/v1/worker/gpu/pp_utils.py"):
    fail.append("#55745 missing: PP draft broadcast still gathers idx_mapping without record_stream")
bw = src("vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py")
pw = src("vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py")
if "does not support decode_context_parallel_size > 1" in src("vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py"):
    fail.append("#50611's push+DCP refusal is still in connector.py")
if "shard identically" not in bw:
    fail.append("push DCP handshake equality check missing")
# #54518 gated _pop_done_transfers on _recving_metadata, which a push producer
# never has, so every WRITE-complete request waited out its 30 s lease (77,584
# warnings in 66850730). The gate must be gone; the recv-side filter that
# replaced it lives in get_finished and does not touch done_sending.
if "if req_id in self._recving_metadata:" in bw:
    fail.append("#54518's push-producer lease regression is back in _pop_done_transfers")
if "done_recving.intersection_update(self._recving_metadata)" not in bw:
    fail.append("recv-side completion filter missing -- _pop_done_transfers shape changed, re-check the push path")
if "_pop_done_transfers(self._sending_transfers)" not in pw:
    fail.append("push worker no longer polls its send transfers the expected way")
if "_align_remote_regions_by_layer" not in bw or "packed_member_layouts" not in bw:
    fail.append("#50499 (rebased head) layer-name routing missing")
refusal = "with Mamba/SSM hybrid KV cache layouts yet" in bw
if "ssm" in ours:
    if "_transfer_layer_ssm_positions" not in bw: fail.append("K3_OURS=ssm but the SSM layer path is not in base_worker")
    if refusal: fail.append("K3_OURS=ssm but the Mamba-under-PP refusal is still present")
elif not refusal:
    fail.append("arm without ssm, yet the Mamba-under-PP refusal is gone -- this tree is not the upstream-only baseline")
mc = src("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py")
if ("mcpp" in ours) != ("set_xfer_handshake_metadata_pp_aware" in mc):
    fail.append("MooncakeStore PP override presence does not match K3_OURS")
if fail: sys.exit("[k3-pp-911] FATAL:\n  - " + "\n  - ".join(fail))
import vllm
print(f"[k3-pp-911] verified: K3_OURS={sorted(ours) or 'none'}; vllm {getattr(vllm, '__version__', '?')}")
PY
echo "=== k3-pp-911: done ==="
