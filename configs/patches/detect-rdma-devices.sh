#!/bin/bash
# shellcheck shell=bash
#
# Fill MOONCAKE_DEVICE and UCX_NET_DEVICES from the node's own inventory.
#
# Not from a config. Wei's file names mlx5_8, which is the HCA on the node he
# ran on; copying it killed four weiport arms today with "Found 5 HCAs /
# mlx5_17 has no active ports" then "Initialize MooncakeDistributedStore
# failed", and the note in tree says an earlier copy of the same value cost six
# runs to "Found 0 HCAs". oci-aga nodes disagree with each other -- today's UCX
# logs showed mlx5_8, mlx5_17, rdma_rail0..3 and rdma_vf_rail0..3 across three
# nodes of the same cluster -- so any hardcoded name is wrong somewhere.
#
# What this is for: the arms that measure 6,818 tok/s/GPU run with
# MOONCAKE_DEVICE unset and UCX_NET_DEVICES unset on a node with five HCAs.
# Mooncake's auto-discovery takes one, and our own comment above warns it takes
# the wrong one on a multi-NIC host. B300 pins eight devices one per GPU; GB300
# pins nothing. Whether that costs anything is unmeasured -- which is the point
# of the arm this feeds.
#
# Sets nothing that is already set: an arm that names devices keeps them.
set -uo pipefail

_have_ports() {  # $1 = device name -> 0 if it has an ACTIVE port
    ibstat "$1" 2>/dev/null | grep -q "State: Active"
}

_detected=""
for _d in $(ibstat -l 2>/dev/null); do
    if _have_ports "$_d"; then
        _detected="${_detected:+${_detected},}${_d}"
    fi
done

if [ -z "${_detected}" ]; then
    echo "[rdma] no ACTIVE RDMA device found; leaving MOONCAKE_DEVICE and" \
         "UCX_NET_DEVICES as they are" >&2
else
    _n=$(echo "${_detected}" | tr ',' '\n' | grep -c .)
    echo "[rdma] ${_n} active device(s): ${_detected}"
    # Mooncake takes a comma list -- nine of the B300 arms already pass one --
    # so give it every active device rather than letting it choose one.
    if [ -z "${MOONCAKE_DEVICE:-}" ]; then
        export MOONCAKE_DEVICE="${_detected}"
        echo "[rdma] MOONCAKE_DEVICE=${MOONCAKE_DEVICE} (was unset)"
    else
        echo "[rdma] MOONCAKE_DEVICE=${MOONCAKE_DEVICE} kept (set by the arm)"
    fi
    # UCX wants port suffixes. Every active device reports port 1 here; if a
    # node ever presents a device whose active port is not 1 this picks the
    # wrong one, so it prints what it built.
    if [ -z "${UCX_NET_DEVICES:-}" ]; then
        _ucx=$(echo "${_detected}" | tr ',' '\n' | sed 's/$/:1/' | paste -sd,)
        export UCX_NET_DEVICES="${_ucx}"
        echo "[rdma] UCX_NET_DEVICES=${UCX_NET_DEVICES} (was unset)"
    else
        echo "[rdma] UCX_NET_DEVICES=${UCX_NET_DEVICES} kept (set by the arm)"
    fi
fi
