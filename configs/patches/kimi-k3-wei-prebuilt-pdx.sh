#!/usr/bin/env bash
# Wei's prebuilt image on aws-pdx: nothing applied, cluster-local checkpoint.
# =============================================================================
# WHY THIS REPLACES kimi-k3-wei-gb300-pdx.sh. The B300 frontier was measured on
# vllm/vllm-openai:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3 plus our
# carried patch. That tag NO LONGER EXISTS. Docker Hub prunes plain
# nightly-<sha> tags -- 18 of them are left in the whole repository and ours is
# not one, so the manifest fetch returns a hard 404 and pyxis fails the step:
#
#   URL https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/
#       nightly-46638857... returned error code: 404 Not Found
#   error: spank: required plugin spank_pyxis.so: task_init() failed
#
# bia kept working only because bia's node cache still holds the layers; it was
# never re-pulled. There is no ssh to bia, so the cache cannot be exported, and
# bia is gone on 2026-09-14 regardless. The image is not recoverable.
#
# WHAT WE MOVE TO, AND WHY IT IS THE RIGHT BASE. Wei's published container
#
#   vllm/vllm-openai:nightly-dev-x86_64-cu13-3696c77          (still live)
#
# is 3696c77, the head of his branch -- the exact commit our 6,368-line patch is
# a diff *of*. So the tree that runs here is the tree the patch was producing;
# the patch is not skipped, it is already in. We measured the pair on bia
# instead of assuming it: our AGG config read 11,915 / 11,834 / 12,845 on his
# image against 12,028 / 11,930 / 12,882 on ours, within 1% at all three
# concurrencies. Frontier numbers reproduced here carry that 1% caveat and
# nothing larger.
#
# It also settles the standing constraint that every config must end up on ONE
# image. This is the one that survives.
#
# THE CHECKPOINT IS ALREADY ON THE CLUSTER, via JET. Rev 9f62e4e -- the one the
# bia frontier ran on -- is synced to aws-pdx-slurm-1_lustre at
#
#   /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/
#     model/moonshotai_kimi-k3/hf/hf-9f62e4e_orig      (96 shards, 1.5 TB)
#
# verified readable from our account. So there is no download: the hfshim
# materialises the standard HF cache layout under $HF_HOME with symlinks.
#
# DO NOT let the shim fall back to its bia default. Its built-in STAGED_DIR is
# /lustre/share/coreai_comparch_aarwlt/hf_repos/... which does not exist here.
# K3_STAGED_DIR is exported below and checked, so a wrong path fails at setup
# rather than an hour into a run.
# =============================================================================
set -euo pipefail

: "${K3_STAGED_DIR:=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/moonshotai_kimi-k3/hf/hf-9f62e4e_orig}"
export K3_STAGED_DIR

# THE SETUP STAGE DOES NOT SEE THE CONFIG'S `environment:` BLOCK. Four runs died
# here with `[k3-hfshim] FATAL: HF_HOME is not set` *after* the shim had already
# confirmed `shards visible: 96` -- the checkpoint was fine, the shim simply had
# nowhere to put the cache entry. The workers get HF_HOME from the YAML; this
# script runs earlier and does not.
#
# So default it, and default it to the SAME path the YAML sets. If these two
# ever drift the shim writes a cache the workers never read, and the failure
# would be a silent re-download rather than an error -- keep them in sync:
#
#   environment.HF_HOME  in  ci/sweep_configs/.../B300/*/**-pdx.yml
#
# An already-set HF_HOME wins, so a caller can still redirect it.
: "${HF_HOME:=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/misunp/hf-cache}"
export HF_HOME
echo "=== wei-prebuilt-pdx: HF_HOME = $HF_HOME ==="
mkdir -p "$HF_HOME" || {
    echo "wei-prebuilt-pdx: FATAL: cannot create HF_HOME at $HF_HOME" >&2
    exit 1
}

echo "=== wei-prebuilt-pdx: staged checkpoint = $K3_STAGED_DIR ==="
if [[ ! -d "$K3_STAGED_DIR" ]]; then
    echo "wei-prebuilt-pdx: FATAL: $K3_STAGED_DIR is not a directory." >&2
    echo "  JET said rev 9f62e4e is synced on aws-pdx-slurm-1_lustre; re-check with" >&2
    echo "  jet_model.py, and note its 'path' subcommand picks the NEWEST artifact" >&2
    echo "  (f831ab66), which is synced nowhere -- query the 9f62e4e artifact by id." >&2
    exit 1
fi
n=$(ls "$K3_STAGED_DIR"/*.safetensors 2>/dev/null | wc -l)
echo "    shards visible: $n"
if [[ "$n" -lt 90 ]]; then
    echo "wei-prebuilt-pdx: FATAL: expected 96 safetensors shards, found $n." >&2
    exit 1
fi

# PROVE HUGE PAGES ARE ON BEFORE SPENDING AN HOUR FINDING OUT THEY ARE NOT.
# EFA counts its registration budget in 4 KiB PAGES (efa_verbs.c:
# max_mr_size = max_mr_pages * PAGE_SIZE), ~383 GiB per device measured here.
# 190 GB x 8 ranks does not fit on 4 KiB pages; the same memory on 2 MiB pages
# does, by 512x.
#
# The segment comes from aligned_alloc inside glibc malloc
# (real_client.cpp -> client_buffer_allocation.cpp), and glibc reaches the
# kernel via its internal __mmap, so an LD_PRELOAD interposer cannot see it --
# three runs were spent proving that the hard way. glibc's own tunable is the
# switch, and the config sets it:
#
#   GLIBC_TUNABLES=glibc.malloc.hugetlb=1
#
# This probe is the gate. It aligned_allocs 2 GiB, touches it, and reads
# AnonHugePages for THAT RANGE out of /proc/self/smaps -- not the global
# counter in /proc/meminfo, which moves for unrelated reasons and is how this
# tunable was first, wrongly, written off.
# NOTE ON WHAT THIS CAN AND CANNOT CHECK. This script runs BEFORE the config's
# `environment:` block is applied -- the same trap that cost four runs with
# HF_HOME -- so GLIBC_TUNABLES is not set here even when the workers will have
# it. Gating on the variable's presence would fail a perfectly good run. So the
# gate proves the MECHANISM works in this container, with and without, and the
# workers' own `Mounting segment` lines are what confirm the plumbing.
echo "=== wei-prebuilt-pdx: verifying huge pages for the store segment ==="
gcc -O2 -o /tmp/thp_probe /configs/patches/thp_probe.c || {
    echo "wei-prebuilt-pdx: FATAL: could not build thp_probe." >&2
    exit 1
}
echo "    control (no tunable, expect 0%):"
/tmp/thp_probe 2 >/dev/null 2>&1 && {
    echo "wei-prebuilt-pdx: NOTE: huge pages are already on without the tunable;" >&2
    echo "  harmless, but it means this container's default differs from pdx's." >&2
}
echo "    arm (glibc.malloc.hugetlb=1, expect 100%):"
if ! GLIBC_TUNABLES=glibc.malloc.hugetlb=1 /tmp/thp_probe 2; then
    echo "wei-prebuilt-pdx: FATAL: the tunable does not produce huge pages here," >&2
    echo "  so the store segment lands on 4 KiB pages, 190 GB x 8 ranks cannot" >&2
    echo "  register on EFA, and every rank dies with 'Failed to mount segment'." >&2
    echo "  glibc must be >= 2.35 (image is 2.39) and THP must not be 'never'." >&2
    exit 1
fi

# The HF cache shim first: a missing checkpoint should fail here, not later.
bash /configs/patches/vllm-container-deps-k3-hfshim.sh

# Then the gate. It applies nothing; it proves the image is his.
bash /configs/patches/kimi-k3-wei-prebuilt.sh

echo "=== wei-prebuilt-pdx: done ==="
