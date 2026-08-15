#!/usr/bin/env bash
# Kimi-K3 DISAGG on the stock vLLM nightly, patched at runtime (GB300).
#
# This replaces kimi-k3-gb300-srcimage-disagg.sh, which needed a 60-minute
# aarch64 image build for every move of wzhao/kimi-k3-agentx-v2 -- and we were
# 139 commits behind it before noticing. Hanjie Qiu reproduced Wei's numbers this
# way on oci-aga, and shared the patch and applier; both are carried verbatim in
# this directory. The vLLM delta is Python only (25 files, no csrc), so nothing
# has to be compiled: the DCP CUDA kernels it depends on are upstream already and
# ship compiled in the nightly.
#
# WHAT THIS DOES NOT CHANGE: the vLLM patch is Hanjie's, byte for byte. Rebasing
# wzhao/kimi-k3-agentx-v2 onto this nightly hits a real semantic conflict --
# upstream grew its own enable_partial_hash_hits and disables it under DCP, while
# Wei's commit is what enables it -- and resolving that ourselves would be
# measuring our merge rather than his stack. His resolution keeps upstream's
# scaffolding, including the unsupported_partial_hit_managers guard, and drops
# only the `dcp_world_size == 1` restriction.
#
# CLUSTER DIFFERENCES, all handled below (this runs on lyris and oci-aga):
#   - NICs are mlx5_0..3, not the mlx5_8 his oci-aga config names. Copying his
#     device_name once already cost us six runs to `Found 0 HCAs`, so this script
#     leaves MOONCAKE_DEVICE unset and lets auto-discovery read the inventory.
#   - the Mooncake segment tops out at 120 GB/rank here, not 150. Wei, 2026-08-13:
#     "150gb global segment size for MC works on OCI-aga, on lyris I can only do
#     120GB". 150 OOM-killed us twice before that was known.
#   - the K3 weights live on a shared path, not in a per-account HF cache.
set -euxo pipefail

export MOONCAKE_VERSION=0.3.12.post1
export FI_VER=${FI_VER:-0.6.16.post3}

# Mooncake host-DRAM segment. vllm-container-deps-mooncake.sh derives the
# per-rank size as TOTAL_CPU_DRAM_GB / MOONCAKE_TP, and MOONCAKE_TP is the ranks
# on this node, so TOTAL_CPU_DRAM_GB is the node-wide reservation: 4 x 120.
#
# Forced, not defaulted. The arms still carry the B300 value in their
# prefill/decode_environment, and a ':-' fallback would silently restore it.
#
# The ceiling is per cluster. Wei, 2026-08-13: "150gb global segment size for MC
# works on OCI-aga, on lyris I can only do 120GB" -- and the trays are the same
# size (950 GB on oci-aga, 955 on lyris), so the difference is not memory and
# cannot be derived here. The arm names it: K3_MOONCAKE_TOTAL_GB.
#
# MEASURED, and it cost a day: 150 GiB/rank does not boot here. At 4 x 120 the
# arms reach "Server is healthy" 18 minutes after the job starts. At 4 x 150 the
# second node of each role stops at "Reducing Torch threads for serving" -- the
# step before KV-cache and connector registration -- and emits nothing for the
# next hour, until the idle-GPU reaper claims it. Four ladders died that way
# before the sweep log was read closely enough to see that the last line was at
# 11:24 and the kill at 12:24.
#
# 600 GB is 63% of the node's 950 GB tray, and registering it through RDMA is
# evidently not something two nodes can finish concurrently here. Wei runs 150 on
# this cluster; his startup cost at that size was never checked before copying
# the number.
#
# There is also no gain to weigh against it today: the offload tier reads back
# 0.0%, so a larger store holds more of what nothing reads. Raise it again only
# with a measurement of both boot time and offload hit rate.
#
# A new name rather than TOTAL_CPU_DRAM_GB, because the arms still carry the
# B300 value under that one and reading it here would silently restore it.
export TOTAL_CPU_DRAM_GB=${K3_MOONCAKE_TOTAL_GB:-480}
export MOONCAKE_TP=4

# Check it against the tray before the run rather than 40 minutes into it. The
# ceiling is bracketed by measurement: 400 GB of a 955 GB tray runs (62405731),
# 600 OOM-kills (62448615), 748 OOM-killed before that (SLURM 2673272).
MEM_TOTAL_GB=$(awk '/^MemTotal:/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
if [ -n "${MEM_TOTAL_GB}" ] && [ "${MEM_TOTAL_GB}" -gt 0 ]; then
    # 55% is the default headroom; an explicit K3_MOONCAKE_TOTAL_GB is trusted
    # up to 70%. The guard exists to catch absurd values -- 748 GB of a 955 GB
    # tray (78%) was OOM-killed -- not to veto a considered one: 600 of 950 is
    # 63%, which is what Wei runs on this cluster.
    _frac=${MOONCAKE_NODE_FRACTION_PCT:-55}
    [ -n "${K3_MOONCAKE_TOTAL_GB:-}" ] && _frac=${MOONCAKE_NODE_FRACTION_PCT:-70}
    MC_CEILING_GB=$(( MEM_TOTAL_GB * _frac / 100 ))
    echo "[mooncake] node MemTotal ${MEM_TOTAL_GB} GB, reserving ${TOTAL_CPU_DRAM_GB} GB" \
         "across ${MOONCAKE_TP} ranks ($(( TOTAL_CPU_DRAM_GB / MOONCAKE_TP )) GB each)," \
         "ceiling ${MC_CEILING_GB} GB"
    if [ "${TOTAL_CPU_DRAM_GB}" -gt "${MC_CEILING_GB}" ]; then
        echo "[mooncake] FATAL: ${TOTAL_CPU_DRAM_GB} GB exceeds ${MC_CEILING_GB} GB." >&2
        exit 1
    fi
else
    echo "[mooncake] WARN: /proc/meminfo unreadable -- cannot check the reservation"
fi

# The staged K3 checkpoint sits at a different path on every cluster, and this
# script now runs on more than one of them, so probe rather than hardcode. An
# explicit K3_STAGED_DIR still wins; otherwise take the first candidate that
# exists and say which. Guessing a path costs a whole run to find out, and the
# team sheet has no OCI-aga entry yet.
if [ -z "${K3_STAGED_DIR:-}" ]; then
    for _cand in \
        /lustre/share/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3 \
        /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/hanjieq/models/kimi-k3 \
        /lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3
    do
        if [ -d "${_cand}" ]; then
            export K3_STAGED_DIR="${_cand}"
            echo "[k3] staged checkpoint: ${K3_STAGED_DIR}"
            break
        fi
    done
fi
if [ -z "${K3_STAGED_DIR:-}" ]; then
    echo "[k3] WARN: no staged checkpoint found on this cluster; the HF shim will" \
         "fall back to downloading 1.45 TB. Set K3_STAGED_DIR to the local copy." >&2
fi

# Optional: fill MOONCAKE_DEVICE and UCX_NET_DEVICES from this node's own
# active RDMA devices. Sourced, not run -- the exports have to survive into
# the worker command, which shares this shell -- and before the deps script,
# which renders MOONCAKE_DEVICE into the mooncake config JSON.
#
# Behind a flag so it changes only the arm that asks for it. Every other arm
# keeps the behaviour that produced the numbers we have.
if [ "${K3_RDMA_AUTOPIN:-0}" = "1" ]; then
    . /configs/patches/detect-rdma-devices.sh
else
    echo "[rdma] autopin off (K3_RDMA_AUTOPIN != 1)"
fi

# hfshim first -- the model has to resolve before anything else matters -- then
# the Mooncake wheel, config and master.
bash /configs/patches/vllm-container-deps-k3-mooncake.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl \
    patch

FI_CUDA=$(python3 -c "
import torch
major = torch.version.cuda.split('.')[0]
print(f'cu{major}' + ('0' if major == '13' else '8'))
")
echo "=== installing flashinfer ${FI_VER} (${FI_CUDA}) ==="
python3 -m pip install --no-deps --force-reinstall "flashinfer-python==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/" "flashinfer-cubin==${FI_VER}"
python3 -m pip install --no-deps --force-reinstall \
    --extra-index-url "https://flashinfer.ai/whl/${FI_CUDA}/" \
    "flashinfer-jit-cache==${FI_VER}+${FI_CUDA}" || \
    echo "WARNING: flashinfer-jit-cache ${FI_VER}+${FI_CUDA} not installed; JIT will compile on demand"


# --- NIXL, shared with the other GB300 nightly setup script ----------------
bash /configs/patches/install-nixl-cu13.sh

# --- UCX / CUDA, shared with the other GB300 nightly setup script ---------
# Both scripts hit the same oci-aga failure, and the last four runs were lost
# to fixing one copy of a duplicated block. There is one copy now.
bash /configs/patches/ucx-cuda-check.sh
SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "site-packages: $SITE"

# Print the root cause LAST. srt-slurm quotes only the final 50 lines of a
# failed process log, and a vLLM engine failure ends with forty lines of
# asyncio/uvloop frames under "Engine core initialization failed. See root
# cause above" -- so the exception that explains it is pushed out of view
# every time. sitecustomize is imported automatically by CPython, so this
# covers every worker without touching a launch command.
cp /configs/patches/k3_roothook.py "${SITE}/k3_roothook.py"
# A .pth, not sitecustomize.py: site.py imports only the first
# sitecustomize on sys.path and the image already ships one at
# /usr/lib/python3.12/sitecustomize.py, which would have shadowed ours
# without a word. A .pth import line runs for every path entry.
echo "import k3_roothook" > "${SITE}/zz-k3-roothook.pth"
# A WARNING, never fatal. The first version of this check ran `import
# sitecustomize` and asserted, which failed on a healthy install and took
# four weiport arms down with it -- the same shape as the UCX guard that
# killed eight arms this morning. A diagnostic that cannot confirm itself
# must not end the run.
if python3 -c "import sys; raise SystemExit(0 if sys.excepthook.__name__ == 'hook' else 1)" 2>/dev/null; then
    echo "[roothook] installed"
else
    echo "[roothook] WARNING: not confirmed active; tracebacks may stay truncated" >&2
fi

# --- Hanjie's applier, unmodified -----------------------------------------
# It gates on _version.py containing g3d204dfda and refuses to touch any other
# image, which is a stronger identity check than we had: our own source-built
# images could not be told apart until v3 forced the issue. Do not soften it --
# the patch is cut against that exact nightly and a near miss applies with
# offsets rather than failing.
export VLLM_SITE_PACKAGES="$SITE"
export VLLM_DCP_PATCH_FILE=/configs/patches/vllm-wzhao-kimi-k3-agentx-v2-on-nightly-aug13.patch
bash /configs/patches/apply-vllm-kimi-k3-dcp-aug13.sh

# --- wzhao18/vllm@face29e659, on top of Hanjie's patch --------------------
# His branch moved one commit after the patch was cut (2026-08-13 14:38,
# "fix(prefix-cache): preserve DSpark PMU replay state"). Kept as its own file
# rather than folded in, so Hanjie's stays byte-for-byte his and our addition is
# reviewable on its own. Verified: applying the two in order reproduces the
# cherry-picked tree exactly.
#
# Inert on these arms -- it populates eagle_proof_units only for groups with
# use_eagle, and every arm here is no-spec. It matters for DSpark/EAGLE, where
# external lookup must prove the same replay boundary the local prefix cache
# does.
EXTRA_PATCH=/configs/patches/vllm-wzhao-face29e659-dspark-pmu-replay.patch
if [ -r "${EXTRA_PATCH}" ]; then
    if patch --batch --forward --dry-run -d "${SITE}" -p1 < "${EXTRA_PATCH}" >/dev/null 2>&1; then
        patch --batch --forward -d "${SITE}" -p1 < "${EXTRA_PATCH}"
        echo "[patch] applied face29e659 (DSpark PMU replay state)"
    elif patch --batch --reverse --dry-run -d "${SITE}" -p1 < "${EXTRA_PATCH}" >/dev/null 2>&1; then
        echo "[patch] face29e659 already present"
    else
        echo "[patch] FATAL: face29e659 neither applies nor is already applied" >&2
        exit 1
    fi
    python3 -m compileall -q "${SITE}/vllm/distributed/kv_transfer" "${SITE}/vllm/v1/core"
else
    echo "[patch] FATAL: ${EXTRA_PATCH} missing" >&2
    exit 1
fi

# --- wzhao18/vllm kimi-k3-agentx-v3, the part we do not already have -------
# Every +dspark arm reached healthy, served traffic, and died in
# cache_full_blocks on `block_hash_num_tokens` -- c8 at 61/87 requests, c32 at
# 82/354, c48 at 123/531. The nospec arms on the identical stack kept running,
# which is the clue: the assertion guards partial->full block promotion, and
# only draft tokens open that window.
#
# 07839fb50f "fix(kv-transfer): privatize partial attention pages" names it:
#
#   A fine-grained local hit can end inside a larger attention page. The page
#   is shared through the prefix cache, so an external suffix must use a newly
#   allocated private page instead of overwriting it in place.
#
# The fine-grained hit is what prefix-match-unit=128 creates, and every arm
# here sets it.
#
# Only four files. His branch is eighteen commits over upstream 925ea7e60f and
# the aug13 snapshot already carries fourteen of them; the delta is this. Cut
# as merge-base..tip -- NOT as our-tree..his-tree, which mixes his changes with
# upstream drift and reads like a dependency on upstream we do not have. That
# misreading cost an hour and two wrong conclusions ("v3 needs #50062", "the
# image must move"). See tools/wei_branch_port.py.
V3_PATCH=/configs/patches/vllm-wzhao-k3-agentx-v3-delta-on-aug13.patch
if [ ! -r "${V3_PATCH}" ]; then
    echo "[patch] FATAL: ${V3_PATCH} missing" >&2
    exit 1
fi
if patch --batch --forward --dry-run -d "${SITE}" -p1 < "${V3_PATCH}" >/dev/null 2>&1; then
    patch --batch --forward -d "${SITE}" -p1 < "${V3_PATCH}"
    echo "[patch] applied v3 delta (privatize partial attention pages, DSpark+DCP)"
elif patch --batch --reverse --dry-run -d "${SITE}" -p1 < "${V3_PATCH}" >/dev/null 2>&1; then
    echo "[patch] v3 delta already present"
else
    echo "[patch] FATAL: the v3 delta neither applies nor is already applied" >&2
    exit 1
fi
# The name the fix hangs on, asserted rather than assumed: without it a +dspark
# arm does not fail here, it fails an hour later under load.
# 07839fb50f is in two halves -- the method, and the nine lines in
# Scheduler.schedule that call it. The port took the first half only: the delta
# was cut per file, sched/scheduler.py came back "previously applied" because
# the v2 snapshot already had most of it, and the one new hunk went with the
# skipped file. Everything compiled, every assertion here passed, and the arm
# died at 122/531 on the assertion the commit exists to prevent -- one request
# off what the unpatched stack managed.
python3 /configs/patches/apply-privatize-external-pages-callsite.py "${SITE}" || exit 1
python3 - <<'V3CHECK' || { echo "[patch] FATAL: the v3 delta did not take" >&2; exit 1; }
import os
import pathlib
site = pathlib.Path(os.environ["VLLM_SITE_PACKAGES"])
src = (site / "vllm/v1/core/kv_cache_manager.py").read_text()
if "truncate_attention_blocks_for_external_load" not in src:
    raise SystemExit("kv_cache_manager has no truncate_attention_blocks_for_external_load")
# A method in the tree is not a method that runs. Assert the caller too.
sched = (site / "vllm/v1/core/sched/scheduler.py").read_text()
if "truncate_attention_blocks_for_external_load" not in sched:
    raise SystemExit("nothing calls truncate_attention_blocks_for_external_load; "
                     "the scheduler half of 07839fb50f is missing")
back = (site / "vllm/v1/attention/backend.py").read_text()
if "supports_non_causal_multi_token_dcp" not in back:
    raise SystemExit("backend.py does not declare supports_non_causal_multi_token_dcp")
print("[patch] v3 delta verified")
V3CHECK

# The one hunk of the v3 port that does not apply: our de19be2460 moved the
# TokenspeedMLA class body, so Wei's context no longer matches. Carried as an
# applier instead of a hunk, because what it sets is read through
# `getattr(builder_cls, "supports_non_causal_multi_token_dcp", False)` -- a
# missing attribute is silently a "no", the selector then refuses
# TOKENSPEED_MLA with "non-causal MLA attention with DCP not supported", every
# worker fails to start, and the job surfaces it as an EOFError twenty minutes
# in. It was first waved through because a neighbouring flag,
# supports_mtp_with_cp_non_trivial_interleave_size, sits a few lines away and
# reads almost the same.
python3 /configs/patches/apply-tokenspeed-noncausal-dcp.py "${SITE}" || exit 1
python3 - <<'DCPCHECK' || { echo "[patch] FATAL: TOKENSPEED_MLA would be refused under DCP" >&2; exit 1; }
import os
import pathlib
site = pathlib.Path(os.environ["VLLM_SITE_PACKAGES"])
ts = (site / "vllm/v1/attention/backends/mla/tokenspeed_mla.py").read_text()
# It has to be on the builder class -- the impl class carries similarly named
# flags and the selector does not read those.
builder = ts.split("class TokenspeedMLAMetadataBuilder")[1].split("\nclass ")[0]
if "supports_non_causal_multi_token_dcp" not in builder:
    raise SystemExit(
        "supports_non_causal_multi_token_dcp is not on TokenspeedMLAMetadataBuilder; "
        "the backend selector will refuse TOKENSPEED_MLA under DCP"
    )
print("[patch] TOKENSPEED_MLA declares non-causal DCP support")
DCPCHECK

# --- ours: block-align the external hit ------------------------------------
# Not from Wei or Hanjie. The mooncake store keys one object per block_size
# chunk, so the only lengths it can serve are multiples of the block size, but
# the lookup aligns to hash_block_size whenever partial hash hits are on. The
# reported hit then lands where nothing was written, the load finds nothing, and
# nothing errors -- which is why every K3 DCP run here writes gigabytes and
# reads back 0.0% external hit while save_put_failed_keys stays 0.
#
# configs/patches/test_mooncake_dcp_keyset.py has been failing on exactly this
# at dcp=2/4/8 (and passing at dcp=1). I dismissed it twice as a stale test
# before connecting it to the 0%.
#
# Reverting this restores the sub-block tail from vllm#49502; measure before
# keeping either way.
# Site 1: how much of a hit is reported. A .patch, cut against this tree.
ALIGN_PATCH=/configs/patches/vllm-mooncake-external-hit-block-aligned.patch
if patch --batch --forward --dry-run -d "${SITE}" -p1 < "${ALIGN_PATCH}" >/dev/null 2>&1; then
    patch --batch --forward -d "${SITE}" -p1 < "${ALIGN_PATCH}"
    echo "[patch] applied: external hit is block-aligned"
elif patch --batch --reverse --dry-run -d "${SITE}" -p1 < "${ALIGN_PATCH}" >/dev/null 2>&1; then
    echo "[patch] external-hit alignment already present"
else
    echo "[patch] FATAL: the external-hit alignment patch does not apply" >&2
    exit 1
fi
# Site 2 and the #50359 revalidation are BOTH request-path changes, and both
# arrived after the last arm that completed. A c48 with them stalls in warmup --
# 51 requests sent, 0 returned in 55 minutes, no errors -- and the idle-GPU
# reaper then claims the job, which is how four ladders looked like a reaper
# problem when they were not.
#
# So they go behind one flag, default off. Off is exactly the stack that measured
# 6,818 tok/s/GPU. K3_MOONCAKE_LOOKUP_PATCHES=1 turns them back on for the arm
# that is testing them.
if [ "${K3_MOONCAKE_LOOKUP_PATCHES:-0}" = "1" ]; then
    python3 /configs/patches/apply-lookup-align.py "${SITE}" || exit 1
else
    echo "[patch] lookup-align skipped (K3_MOONCAKE_LOOKUP_PATCHES != 1)"
fi
# And one sample key from each side. The master says 19,224 keys are resident
# and Get never fires, so written and queried keys disagree; a key is a string,
# so print one of each and read the difference instead of deriving it.
python3 /configs/patches/apply-keysample-log.py "${SITE}" || exit 1
# vllm#50359, as we carried it on misunp/k3-merged-mooncake-dcp and then left
# behind when this track moved to the nightly. It revalidates that a reported
# partial-hash hit names an object the store actually holds, and shortens by
# one alignment unit until it does -- "a longer stored key proves one object
# exists at that endpoint; it does not imply independently addressable objects
# at every shorter hash boundary". That is the shape of our 0.0%: the lookup
# asks about 240,000 keys per window, reports no error, and never issues a
# Get. Measured on B300 c8 DCP=8 before: 2,757,664 OBJECT_NOT_FOUND.
#
# Cut against the real target tree (nightly 3d204dfda + Hanjie + face29e659 +
# our alignment work), not against repo/vllm -- applying the branch commit
# directly lands with fuzz 2 and a 44-line offset.
REVAL_PATCH=/configs/patches/vllm-mooncake-50359-exact-hit-revalidate.patch
if [ "${K3_MOONCAKE_LOOKUP_PATCHES:-0}" != "1" ]; then
    echo "[patch] #50359 revalidation skipped (K3_MOONCAKE_LOOKUP_PATCHES != 1)"
elif patch --batch --forward --dry-run -d "${SITE}" -p1 < "${REVAL_PATCH}" >/dev/null 2>&1; then
    patch --batch --forward -d "${SITE}" -p1 < "${REVAL_PATCH}"
    echo "[patch] applied: #50359 exact-hit revalidation"
elif patch --batch --reverse --dry-run -d "${SITE}" -p1 < "${REVAL_PATCH}" >/dev/null 2>&1; then
    echo "[patch] #50359 revalidation already present"
else
    echo "[patch] FATAL: the #50359 revalidation patch does not apply" >&2
    exit 1
fi
python3 -m compileall -q "${SITE}/vllm/distributed/kv_transfer"
# compileall proves the syntax, not that the module runs. Today a patched
# coordinator.py compiled fine and raised NameError on `logger` at the
# first lookup -- inside a worker thread, so nothing crashed and the
# client hung at "0 returned, 0 errors" until the reaper killed the job.
# Import the modules we patched and touch the names we added.
# Ask the module what it uses, do not assume. The first version of this check
# required coordinator.py to define a logger unconditionally -- but the logger
# arrives with apply-lookup-align.py, which is gated off by default, so the
# guard failed every arm that ran the default configuration and passed only the
# ones testing the patch. Four arms died at eight minutes to a check written to
# prevent a different bug. A guard that fires when nothing is wrong is worse
# than no guard: it teaches you to route around it.
python3 - <<'IMPORTCHECK' || { echo "[patch] FATAL: a patched module does not import cleanly" >&2; exit 1; }
import importlib
import pathlib

for m in ("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator",
          "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker"):
    mod = importlib.import_module(m)
    src = pathlib.Path(mod.__file__).read_text()
    if "logger." in src and not hasattr(mod, "logger"):
        raise SystemExit(f"{m} logs but defines no logger")
print("[patch] patched modules import and expose what they use")
IMPORTCHECK

echo "=== verifying the patched tree ==="
# Read the tree rather than trust the applier's exit code: `patch --forward`
# reports success when it decides a hunk is already applied, so a partially
# stale tree can exit 0. These assert the behaviours the arms depend on.
python3 -c "
import ast
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
read = lambda rel: (root / rel).read_text()

coord = read('v1/core/kv_cache_coordinator.py')
assert 'def get_replay_boundary' in coord, (
    'the patch did not land: no get_replay_boundary'
)
# The conflict Hanjie resolved: upstream disables fine-grained hits under DCP,
# Wei's change is what enables them. Every arm here is DCP8, so a silent revert
# to the upstream form would turn prefix-match-unit off and look like a plain
# regression.
assert 'enable_partial_hash_hits = has_partial_mamba_group' in coord, (
    'partial hash hits are still gated on dcp_world_size == 1: the DCP arms '
    'would run with prefix-match-unit silently inert'
)
assert 'dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size' in coord, (
    'the DCP branch of has_partial_mamba_group is missing'
)

proof = read('distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py')
assert 'eagle_proof_units' in proof, (
    'wzhao18/vllm@face29e659 did not land: external EAGLE lookup will not prove '
    'the replay boundary the local prefix cache enforces'
)

stkcm = read('v1/core/single_type_kv_cache_manager.py')
assert '(block_idx, req_blocks[block_idx])' in stkcm, (
    'the partial-hit CoW still records new_computed_blocks[-1]; '
    'allocate_new_blocks dies on a bare AssertionError under load'
)

# vllm#51733, the MLA prefill workspace fix Wei named as the CUDA-OOM guard.
# It is upstream and therefore in the nightly, but it is the one thing whose
# absence explains every weicfg arm we lost, so check rather than assume.
mla = read('model_executor/layers/attention/mla_attention.py')
assert 'max_num_seqs' not in mla.split('def align_mla_chunked_context_workspace_size')[1].split('def ')[0], (
    'MLA prefill workspace still scales with max_num_seqs: vllm#51733 is absent '
    'and the prefill worker will OOM at gmu 0.92'
)
print('=== patched tree verified ===')
"

echo "=== mooncake client/master version agreement ==="
python3 - <<'MCGATE'
import importlib.metadata as md
import os
import sys

want = os.environ.get("MOONCAKE_VERSION", "0.3.11.post1")
try:
    import mooncake
    from mooncake.store import MooncakeDistributedStore  # noqa: F401
except Exception as exc:
    sys.exit("mooncake is not importable after setup: %s" % exc)

dists = [d for d in md.distributions()
         if (d.metadata["Name"] or "").startswith("mooncake-transfer-engine")]
present = sorted("%s==%s" % (d.metadata["Name"], d.version) for d in dists)
print("  imported from: %s" % getattr(mooncake, "__file__", "?"))
print("  distributions present: %s" % (present or "none"))
if len(dists) != 1:
    sys.exit("expected exactly one mooncake-transfer-engine distribution, found "
             "%d: %s" % (len(dists), present))
if dists[0].version != want:
    sys.exit("mooncake is %s, not the pinned %s -- the deps script fell back"
             % (present[0], want))
print("  ok: %s" % present[0])
MCGATE

echo "=== GB300 nightly disagg setup complete ==="
