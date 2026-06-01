# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the data-parallel-rpc-port per-replica offset fix.

When multiple DP-EP prefill (or decode) replicas share a node — common on
8-GPU/node B300 with DEP=4 replicas (2 reps/node) — every replica was
defaulting to the same TCP port for its DP master (data-parallel-rpc-port,
typically 13346), causing `zmq.error.ZMQError: Address already in use` and
killing the second replica's engine startup.

Fix: srtctl's vLLM backend now adds the Process.endpoint_index to the
configured base port, so each replica binds a unique port on its node.
"""

from srtctl.core.topology import (
    allocate_endpoints,
    endpoints_to_processes,
)


def test_endpoint_index_unique_per_replica_on_shared_node():
    """Two DEP=4 replicas on the same node must have different endpoint_index."""
    endpoints = allocate_endpoints(
        num_prefill=2,
        num_decode=0,
        num_agg=0,
        gpus_per_prefill=4,
        gpus_per_decode=8,
        gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001",),
    )
    processes = endpoints_to_processes(endpoints)
    # Both replicas land on bia0001 (4 GPU each, 8 GPU/node).
    nodes_seen = {p.node for p in processes}
    assert nodes_seen == {"bia0001"}, f"expected both replicas on bia0001, got {nodes_seen}"
    # Two distinct endpoint_index values, both DP-leader (node_rank=0).
    indices = sorted({p.endpoint_index for p in processes})
    assert indices == [0, 1], f"expected endpoint_index 0 and 1, got {indices}"


def test_endpoint_index_offset_yields_unique_ports():
    """Simulate the patch: base + endpoint_index → unique per replica."""
    BASE = 13346
    endpoints = allocate_endpoints(
        num_prefill=4,
        num_decode=0,
        num_agg=0,
        gpus_per_prefill=4,
        gpus_per_decode=8,
        gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001", "bia0002"),
    )
    processes = endpoints_to_processes(endpoints)
    # 4 dep4 replicas across 2 nodes (2 per node). Verify ports are unique
    # WITHIN each node (across-node collisions are fine — different IPs).
    per_node = {}
    for p in processes:
        per_node.setdefault(p.node, []).append(BASE + p.endpoint_index)
    for node, ports in per_node.items():
        assert len(set(ports)) == len(ports), (
            f"port collision on {node}: {ports}"
        )


def test_eight_replicas_two_per_node_no_port_collision():
    """8 dep4 replicas on 4 nodes (2/node): all 8 distinct ports, no per-node collision."""
    BASE = 13346
    endpoints = allocate_endpoints(
        num_prefill=8,
        num_decode=0,
        num_agg=0,
        gpus_per_prefill=4,
        gpus_per_decode=8,
        gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001", "bia0002", "bia0003", "bia0004"),
    )
    processes = endpoints_to_processes(endpoints)
    # Group ports per node — both replicas on a node must have distinct ports.
    per_node = {}
    for p in processes:
        per_node.setdefault(p.node, []).append(BASE + p.endpoint_index)
    for node, ports in per_node.items():
        assert len(ports) == 2, f"expected 2 replicas on {node}, got {len(ports)}"
        assert len(set(ports)) == 2, f"port collision on {node}: {ports}"
    # 8 globally distinct ports overall (offset is unique).
    all_ports = [BASE + p.endpoint_index for p in processes]
    assert len(set(all_ports)) == 8, f"expected 8 unique ports, got {sorted(set(all_ports))}"
