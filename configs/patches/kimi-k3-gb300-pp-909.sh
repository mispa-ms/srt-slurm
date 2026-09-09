#!/usr/bin/env bash
# Kimi-K3 GB300 PP + disagg on the 2026-09-09 nightly.
# =============================================================================
# BASE. vllm/vllm-openai:nightly-385dce36bcee42309924a5ece951a96db3dce7f2
# (upstream 385dce36bc, 2026-09-09 04:53 UTC; arm64 + amd64 on the hub).
#
# PURPOSE. Reproduce the GB300 PP2 disagg frontier on the newest nightly with
# the smallest chain, and run our carries in the form they will be PR'd.
# Same shape as kimi-k3-gb300-pp-908.sh; differences from the 09-08 chain:
#
#  - ckptidx-829 (= vllm#55924) and dspark-pr55472 (= vllm#55472) are MERGED in
#    this nightly. Their steps stay and take the "already present" exits.
#  - pr50499-909 replaces pr50499-908: #50499 was refreshed 2026-09-09 (head
#    0f6c6aa68f, wire v13, per-region geometry on top of merged #53780). The
#    09-08 patch no longer applies. Cut as `git diff c268198715 0f6c6aa68f -- vllm/`.
#  - ssm-909 replaces ssm-908: our SSM member path re-derived on that head
#    (per-member block counts). vllm misunp/k3-ssm-members-on-50499-0f6c6aa.
#  - pr55900-908 is DROPPED. #55900 was closed unmerged (superseded by #56104,
#    which does not apply here either), and neither is needed: the refreshed
#    #50499 carries zixi-qi's 041018a697 "Preserve completion reporting for
#    push writes" (_pop_done_transfers(..., is_send=True)), so the #54518
#    push-producer lease regression (77,584 expired leases in 66850730) is gone.
#  - Everything else is the 09-08 chain verbatim: hfshim, draft-loader guard
#    (#54416 still unmerged; anchors present), pushdcp-908, mcpp-908, pr55745-908
#    (#55745 still unmerged; applies clean).
#
# UPSTREAM-ONLY, EVERY ARM, IN ORDER: hfshim -> ckptidx-829 (no-op) ->
# dspark-draft-loader -> dspark-pr55472 (no-op) -> pr50499-909 -> pushdcp-908
# -> [ssm-909] -> [mcpp-908] -> pr55745-908.
#
# OURS, BY K3_OURS:
#  ssm  -- SSM/Mamba members over the member-identity path (necessary: the
#          09-08 A/B arm A died at the Mamba-under-PP refusal, 66840854).
#  mcpp -- MooncakeStore PP handshake override (necessary with a mooncake tier:
#          0908bmc died at 'received pp_rank > 0 handshake metadata').
#
# Replayed 2026-09-09 on a pristine 385dce36 tree: every patch applies in this
# order; the verify block below passes with K3_OURS=ssm and with K3_OURS=ssm,mcpp.
# =============================================================================
set -euo pipefail
K3_OURS="${K3_OURS:-}"
echo "[k3-pp-909] K3_OURS='${K3_OURS}'"
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
bash /configs/patches/vllm-container-deps-k3-pr50499-909.sh
bash /configs/patches/vllm-container-deps-k3-pushdcp-908.sh
case ",${K3_OURS}," in *,ssm,*)  bash /configs/patches/vllm-container-deps-k3-ssm-909.sh ;; esac
case ",${K3_OURS}," in *,mcpp,*) bash /configs/patches/vllm-container-deps-k3-mcpp-908.sh ;; esac
bash /configs/patches/vllm-container-deps-k3-pr55745-908.sh
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
bw = src("vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py")
if "does not support decode_context_parallel_size > 1" in src("vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py"):
    fail.append("#50611's push+DCP refusal is still in connector.py")
if "shard identically" not in bw:
    fail.append("push DCP handshake equality check missing")
pw = src("vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py")
if "idx_mapping.record_stream(self.broadcast_stream)" not in src("vllm/v1/worker/gpu/pp_utils.py"):
    fail.append("#55745 missing: PP draft broadcast still gathers idx_mapping without record_stream (wall 3 race)")
if "is_send=True" not in pw or "if req_id in self._recving_metadata:" in bw:
    fail.append("push producer done_sending not preserved (#50499 041018a697 missing) -- every request would expire its lease")
if "_align_remote_regions_by_member" not in bw or "packed_member_layouts" not in bw:
    fail.append("#50499 (refreshed head) member-identity routing missing")
refusal = "with Mamba/SSM hybrid KV cache layouts yet" in bw
if "ssm" in ours:
    if "_member_ssm_positions" not in bw: fail.append("K3_OURS=ssm but the SSM member path is not in base_worker")
    if refusal: fail.append("K3_OURS=ssm but the Mamba-under-PP refusal is still present")
elif not refusal:
    fail.append("arm without ssm, yet the Mamba-under-PP refusal is gone -- this tree is not the upstream-only baseline")
mc = src("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py")
if ("mcpp" in ours) != ("set_xfer_handshake_metadata_pp_aware" in mc):
    fail.append("MooncakeStore PP override presence does not match K3_OURS")
if fail: sys.exit("[k3-pp-909] FATAL:\n  - " + "\n  - ".join(fail))
import vllm
print(f"[k3-pp-909] verified: K3_OURS={sorted(ours) or 'none'}; vllm {getattr(vllm, '__version__', '?')}")
PY
echo "=== k3-pp-909: done ==="
