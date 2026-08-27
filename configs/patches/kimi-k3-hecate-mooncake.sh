#!/usr/bin/env bash
# Kimi-K3 on Hecate / VR200 with the Mooncake KV store over RDMA.
#
# ── WHY THIS SCRIPT EXISTS ───────────────────────────────────────────────────
# The probe reported, on every Hecate run so far:
#     WARNING: ibv_devices unavailable (no RDMA visible in container)
# That is not the fabric missing. nvidia-smi topo sees eight HCAs on this node --
# NIC0..7 = mlx5_0, mlx5_1, mlx5_2, mlx5_5, mlx5_6, mlx5_7, mlx5_8, mlx5_11 --
# so the hardware and the kernel side are there. What is absent is the ibverbs
# USERSPACE inside the container image.
#
# This is a known shape: the AIB aarch64 image build had the same problem and it
# was fixed by adding rdma-core / libibverbs1 / ibverbs-providers to the
# Dockerfile. We cannot rebuild the framework team's Rubin image, so we install
# the same packages at job start instead.
#
# ── WHY NOT protocol: tcp ────────────────────────────────────────────────────
# It is the documented fallback and it works, but it is not what the B300 v7
# ladder measured, and its ports run out: without
# MC_TCP_ENABLE_CONNECTION_POOL the tcp transport opens a socket per transfer
# and exhausts the ephemeral range about an hour in, which is exactly the length
# of every point here. An arm that has to be nursed like that is not the arm we
# want on a frontier chart. RDMA, or say plainly that the tier did not run.
#
# ── THE FAILURE THIS GUARDS AGAINST ──────────────────────────────────────────
# Mooncake's worst failure mode is silent. With no device_name it turns on auto
# discovery, finds nothing, mounts no segment, and offloads ZERO bytes while the
# server comes up healthy and serves traffic. The run then looks like "mooncake
# is no better than nothing" when mooncake never ran at all. So this script
# FAILS THE JOB if ibv_devices is still empty after the install, rather than
# letting the arm produce a number nobody can interpret.

set -euo pipefail

bash /configs/patches/kimi-k3-hecate-probe.sh

echo "=== installing the ibverbs userspace Mooncake needs ==="
apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    rdma-core libibverbs1 ibverbs-providers ibverbs-utils

echo "=== RDMA devices now visible in the container ==="
if ! ibv_devices; then
    echo "FATAL: ibv_devices still fails after installing rdma-core."
    echo "       Mooncake would fall back to auto-discovery, mount no segment,"
    echo "       and report a healthy server with 0% offload. Refusing to run."
    exit 1
fi

FOUND=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1 != "" {print $1}' | sort -u | tr '\n' ',' | sed 's/,$//')
if [ -z "$FOUND" ]; then
    echo "FATAL: ibv_devices ran but listed no HCAs."
    echo "       nvidia-smi topo sees NIC0..7 on this node, so the devices are"
    echo "       not exposed into the container. Mooncake cannot use RDMA here."
    exit 1
fi
echo "  usable HCAs: ${FOUND}"

# The config names its devices explicitly; check the two agree, because naming
# a device the container cannot open is how an arm gets a silent 0% offload.
WANT="${K3_MOONCAKE_DEVICES:-}"
if [ -n "$WANT" ]; then
    echo "  config asks for: ${WANT}"
    MISSING=""
    for d in $(echo "$WANT" | tr ',' ' '); do
        echo "$FOUND" | tr ',' '\n' | grep -qx "$d" || MISSING="$MISSING $d"
    done
    if [ -n "$MISSING" ]; then
        echo "FATAL: these device_name entries are not present:${MISSING}"
        echo "       Present:  ${FOUND}"
        echo "       Fix device_name in the config rather than letting Mooncake"
        echo "       silently mount nothing."
        exit 1
    fi
    echo "  all requested devices are present"
fi

echo "=== RDMA userspace ready ==="
echo "NOTE: read the worker log for 'Transfer engine auto discovery is disabled'"
echo "      and the master's Clients count before trusting any offload number."
