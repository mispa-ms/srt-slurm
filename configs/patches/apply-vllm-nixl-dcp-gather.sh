#!/usr/bin/env bash
# NIXL DCP gather for MLA at any local TP -- lift the tp_ratio == dcp_ratio guard.
# =============================================================================
# WHAT IT UNBLOCKS. With decode DCP=1 the v2 patch requires
#     prefill_TP / decode_TP == prefill_DCP / decode_DCP
# so prefill TP8/DCP8 admits exactly one decode TP, and it is 1. Every wide-EP
# decode topology that shards the dense projections needs decode TP > 1, so the
# only way to reach one was to drop prefill DCP in step -- which costs prefill KV
# capacity: DCP8 gives a 20,709,376-token pool, DCP2 gives 5,982,708. At the
# operating point this study runs (64 x 131,072 = 8,388,608 tokens of prefix)
# the small pool cannot hold the working set, the seeded prefixes are evicted
# before the replay, and TTFT p90 goes 10.2 s -> 46.2 s. That is not a
# measurement artefact; a deployment of that topology would thrash the same way.
#
# WHY THE GUARD WAS THERE, and why it is not fundamental. The old body was
#     start = tp_rank * dcp_ratio
#     attn_ranks = list(range(start, start + dcp_ratio))
# which walks a *disjoint* window of remote ranks per local rank. That only
# stays in range when dcp_ratio <= tp_ratio, hence the assert -- and it was only
# ever correct because the assert forced local TP to 1, making tp_rank always 0.
#
# MLA does not want disjoint windows. The latent KV is duplicated across remote
# TP (the function's own comment: "For MLA, we only need one remote since cache
# is duplicated") and sharded by token across remote DCP. The consumer pairs
# them in base_worker._map_dcp_attention_block_ids with
#     dcp_rank = remote_rank % remote_dcp_size
#     local_block_ids[i] = group[dcp_rank :: remote_dcp_size]
# so a local rank reconstructs the sequence iff the residues of its source ranks
# mod remote_dcp_size are exactly {0 .. D-1}. Any D consecutive remote ranks do
# that, for every local rank. Which set it takes is free, so pick it by tp_rank
# and spread readers across the duplicate sets.
#
# The two shapes that worked before produce byte-identical rank lists; three
# that raised now resolve. The verifier below checks that, on the patched tree,
# by value.
#
# NOT SUFFICIENT ON ITS OWN. A wrong mapping does not crash -- it delivers
# mis-paired KV and the perf numbers stay plausible while the text turns to
# noise. This harness runs with ignore_eos and carries no accuracy signal, so
# ship this only alongside a GSM8K arm on the same stack.
# =============================================================================
set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly PATCH_FILE="${VLLM_NIXL_DCP_PATCH_FILE:-/configs/patches/vllm-nixl-dcp-gather-any-tp.patch}"
readonly MARKER_FILE="${VLLM_ROOT}/.nixl_dcp_gather_any_tp"
readonly TARGET="${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py"

if [[ -f "${MARKER_FILE}" ]]; then
  echo "[nixl-dcp] already applied."
  exit 0
fi
if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "[nixl-dcp] FATAL: missing patch ${PATCH_FILE}" >&2
  exit 1
fi

# This patches the *v2-patched* form: the DCP block it edits is added by
# vllm-wzhao-kimi-k3-agentx-v2-on-nightly-*.patch. Applying it to a stock tree
# would fail, so say which ordering is wrong rather than leaving a rejected hunk.
if ! grep -q "remote_dcp_size" "${TARGET}"; then
  echo "[nixl-dcp] FATAL: ${TARGET} has no remote_dcp_size." >&2
  echo "[nixl-dcp] Run the Kimi-K3 agentx-v2 applier first; this patch sits on top of it." >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  echo "[nixl-dcp] content already present."
else
  echo "[nixl-dcp] FATAL: patch neither applies nor is already applied." >&2
  exit 1
fi

python3 -m compileall -q "${TARGET}"

# Assert values, not presence. The pr50359 verifier asked hasattr of an
# attribute set in __init__ -- always False on the class -- and killed three
# arms at startup before anyone noticed it could not pass. Call the real
# function on the patched tree and compare the rank lists.
python3 - <<'PY'
import sys
from dataclasses import dataclass
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import compute_tp_mapping

@dataclass
class Topo:
    tp_rank: int
    tp_size: int
    is_mla: bool = True
    total_num_kv_heads: int = 1

def ranks(tp_rank, tp_size, remote_tp, remote_dcp):
    m = compute_tp_mapping(
        Topo(tp_rank, tp_size), remote_tp, (),
        remote_dcp_size=remote_dcp, local_dcp_size=1,
    )
    for name in ("attn_ranks", "attention_ranks", "source_ranks"):
        if hasattr(m, name):
            return list(getattr(m, name))
    raise AssertionError(f"TPMapping exposes none of the expected rank fields: {dir(m)}")

# (remote_tp, remote_dcp, local_tp) -> expected list per local rank
CASES = {
    (8, 8, 1): [[0, 1, 2, 3, 4, 5, 6, 7]],                                   # unchanged
    (8, 2, 4): [[0, 1], [2, 3], [4, 5], [6, 7]],                             # unchanged
    (8, 8, 4): [[0, 1, 2, 3, 4, 5, 6, 7]] * 4,                               # newly allowed
    (8, 4, 4): [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]],     # newly allowed
}
bad = 0
for (rtp, rdcp, ltp), expected in CASES.items():
    for tp_rank, want in enumerate(expected):
        got = ranks(tp_rank, ltp, rtp, rdcp)
        ok = got == want
        bad += 0 if ok else 1
        print(f"[nixl-dcp] P TP{rtp}/DCP{rdcp} -> D TP{ltp} rank {tp_rank}: "
              f"{'ok ' if ok else 'BAD'} {got}" + ("" if ok else f" != {want}"))
        # the invariant the consumer needs, checked independently of the expectation
        res = sorted(r % rdcp for r in got)
        if res != list(range(rdcp)):
            bad += 1
            print(f"[nixl-dcp] BAD residues {res}, need {list(range(rdcp))}")
if bad:
    print(f"[nixl-dcp] FATAL: {bad} mismatch(es)", file=sys.stderr)
    sys.exit(1)
print("[nixl-dcp] verifier: all rank sets correct")
PY

touch "${MARKER_FILE}"
echo "[nixl-dcp] applied: MLA DCP gather now works at any local TP."
