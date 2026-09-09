#!/usr/bin/env bash
# Kimi-K3 GB300 PP + disagg on the 2026-09-08 nightly -- the A/B stack.
# =============================================================================
# BASE. vllm/vllm-openai:nightly-9ea8f3ffc354901b740f0b31988900897b7221d7
# (upstream 9ea8f3ffc3, 2026-09-08 06:16 UTC).
#
# PURPOSE. Decide which of our carries are still necessary on a current tree.
# Every arm gets the upstream-only stack below; K3_OURS (comma list) adds ours.
#
# UPSTREAM-ONLY, EVERY ARM, IN ORDER:
#  1. hfshim -- HF cache -> staged checkpoint.
#  2. ckptidx-829 -- the int64 checkpoint index. Byte-identical to vllm#55924
#     (opened 2026-09-08); patch(1) absorbs the +156 line offset. Not ours.
#  3. revert52388-829 -- vllm#53774, still open; its crash site is still at
#     mamba_utils.py:1083 ('expected N block tables').
#  4. pr50499-908 -- vllm#50494 + #50499 head ce830b04 (2026-09-05). Applies
#     clean. Fixes the block_strides remap itself, so that carry is gone.
#  5. dspark-draft-loader -- #54416's mechanism (its patch does not apply to
#     this nightly): under PP the drafter lives on the last stage only, but
#     fastsafetensors collectives over group.WORLD, so the draft load deadlocks.
#     Our configs set load-format: fastsafetensors on both roles.
#  6. dspark-pr55472 -- vllm#55472 verbatim: merged #50514 hands the draft a
#     parallel config without DCP, and the decode side (TP8 x DCP8) dies in
#     profile_cudagraph_memory on `assert isinstance(self.dcp_manager,
#     MLADCPManager)`. Pipeline 66821387 (both arms) died exactly there.
#     Steps 5 and 6 are the other session's B200 stopgaps, proven green on this
#     nightly in pipelines 66814090 / 66814200. Loader guard MUST precede #55472.
#  7. pushdcp-908 -- lift #50611's blanket refusals of push + DCP and hybrid +
#     DCP for the push path only; enforce equal DCP at handshake. Pipeline
#     66827142 (both arms) died on decode at NixlPushConnector.__init__:
#     "does not support decode_context_parallel_size > 1". Ours, but a carry
#     for the base: without it no K3 disagg topology exists on this nightly.
#  8. pr55900-908 -- vllm#55900 verbatim, LAST (applies at offsets on top of
#     the other base_worker patches). Undoes #54518's push-producer regression:
#     without it every WRITE-complete request still waits out the 30 s lease
#     (77,584 "Releasing expired KV blocks" lines in 66850730, 0 on 08-29).
#
# OURS, BY K3_OURS:
#  ssm  -- SSM/Mamba members over the member-identity path. #50499 says
#          "Mamba/SSM hybrid layouts under PP remain unsupported" and refuses
#          at worker init; K3 is a Mamba hybrid, so arm A is expected to die
#          there. Unit-tested on CPU (test_nixl_push_connector.py hybrid tests,
#          test_nixl_connector_hma.py PP2 register test).
#  mcpp -- MooncakeStore PP handshake override. Only matters with a mooncake
#          tier; the nomc arms never reach the guard.
#
# DROPPED FROM THE 08-29 CHAIN, AND WHY:
#  - #53803 retention: draft, mergify conflicts since 08-29, author defers to
#    #54076 and expects to drop the rolling half. main has its own internal
#    checkpoints (#52789, #53614). Measure prefix hit on main's mechanism.
#  - k3-engine-0829.patch: 38 hunks reject on this nightly. #50514 merged
#    (dspark_mla layer numbering is upstream verbatim), not_finishing was
#    deleted upstream (6bafc049aa), block_strides is fixed in #50499 itself.
#  - Everything decode-PP (write_ranks fan-out, decode_pp_size, pull refusal):
#    not exercised by P_PP2 -> D_PP1 and push_worker was rewritten upstream.
# =============================================================================
set -euo pipefail
K3_OURS="${K3_OURS:-}"
echo "[k3-pp-908] K3_OURS='${K3_OURS}'"
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
bash /configs/patches/vllm-container-deps-k3-revert52388-829.sh
bash /configs/patches/vllm-container-deps-k3-dspark-draft-loader.sh
bash /configs/patches/vllm-container-deps-k3-dspark-pr55472.sh
bash /configs/patches/vllm-container-deps-k3-pr50499-908.sh
bash /configs/patches/vllm-container-deps-k3-pushdcp-908.sh
case ",${K3_OURS}," in *,ssm,*)  bash /configs/patches/vllm-container-deps-k3-ssm-908.sh ;; esac
case ",${K3_OURS}," in *,mcpp,*) bash /configs/patches/vllm-container-deps-k3-mcpp-908.sh ;; esac
bash /configs/patches/vllm-container-deps-k3-pr55900-908.sh
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
    fail.append("int64 checkpoint index missing")
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
if "is_recv=False" not in pw or "_failed_recv_pending" not in bw:
    fail.append("#55900 missing: push producer never reports done_sending (every request expires its lease)")
if "_align_remote_regions_by_member" not in bw:
    fail.append("#50499 member-identity routing missing")
refusal = "with Mamba/SSM hybrid KV cache layouts yet" in bw
if "ssm" in ours:
    if "_member_ssm_positions" not in bw: fail.append("K3_OURS=ssm but the SSM member path is not in base_worker")
    if refusal: fail.append("K3_OURS=ssm but the Mamba-under-PP refusal is still present")
elif not refusal:
    fail.append("arm without ssm, yet the Mamba-under-PP refusal is gone -- this tree is not the upstream-only baseline")
mc = src("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py")
if ("mcpp" in ours) != ("set_xfer_handshake_metadata_pp_aware" in mc):
    fail.append("MooncakeStore PP override presence does not match K3_OURS")
if fail: sys.exit("[k3-pp-908] FATAL:\n  - " + "\n  - ".join(fail))
import vllm
print(f"[k3-pp-908] verified: K3_OURS={sorted(ours) or 'none'}; vllm {getattr(vllm, '__version__', '?')}")
PY
echo "=== k3-pp-908: done ==="
