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


# ─────────────────────────────────────────────────────────────────
# NIXL handshake-listener port collision (the SECOND bug observed
# after the data-parallel-rpc-port fix: same node, 2 replicas, each
# vLLM internally binds base + stride * dp_rank for the NIXL
# handshake listener — stride observed = 2 — so consecutive 1-port-
# per-rank allocation overlaps between replicas)
# ─────────────────────────────────────────────────────────────────


def _backend(dp_size_p: int, dp_size_d: int):
    from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
    return VLLMProtocol(
        connector="nixl",
        vllm_config=VLLMServerConfig(
            prefill={"data-parallel-size": dp_size_p},
            decode={"data-parallel-size": dp_size_d},
        ),
    )


def test_nixl_port_same_within_replica():
    """All DP ranks of one replica share the same VLLM_NIXL_SIDE_CHANNEL_PORT base."""
    eps = allocate_endpoints(
        num_prefill=2, num_decode=0, num_agg=0,
        gpus_per_prefill=4, gpus_per_decode=8, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001",),
    )
    procs = _backend(4, 8).endpoints_to_processes(eps)
    by_replica = {}
    for p in procs:
        by_replica.setdefault(p.endpoint_index, set()).add(p.nixl_port)
    for idx, ports in by_replica.items():
        assert len(ports) == 1, f"replica {idx} has multiple nixl_port values: {ports}"


def test_nixl_port_gap_between_replicas_dp4():
    """DP=4 replicas need >=8 port gap (dp_size*2) so vLLM's stride=2 fits without overlap."""
    eps = allocate_endpoints(
        num_prefill=8, num_decode=0, num_agg=0,
        gpus_per_prefill=4, gpus_per_decode=8, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001", "bia0002", "bia0003", "bia0004"),
    )
    procs = _backend(4, 8).endpoints_to_processes(eps)
    bases = sorted({p.nixl_port for p in procs})
    assert len(bases) == 8, f"expected 8 distinct replica bases, got {bases}"
    gaps = [b - a for a, b in zip(bases, bases[1:])]
    assert all(g >= 8 for g in gaps), f"some replica-to-replica gap < 8: gaps={gaps}"


def test_nixl_port_gap_between_replicas_dp8():
    """DP=8 replicas need >=16 port gap."""
    eps = allocate_endpoints(
        num_prefill=0, num_decode=2, num_agg=0,
        gpus_per_prefill=8, gpus_per_decode=8, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001", "bia0002"),
    )
    procs = _backend(8, 8).endpoints_to_processes(eps)
    bases = sorted({p.nixl_port for p in procs})
    assert len(bases) == 2
    assert bases[1] - bases[0] >= 16, f"DP=8 gap too small: {bases}"


def test_no_replica_port_block_overlap_at_stride_2():
    """Simulate vLLM stride=2 per rank; reserved blocks per replica must not overlap."""
    eps = allocate_endpoints(
        num_prefill=7, num_decode=1, num_agg=0,
        gpus_per_prefill=4, gpus_per_decode=8, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001", "bia0002", "bia0003", "bia0004", "bia0005"),
    )
    procs = _backend(4, 8).endpoints_to_processes(eps)
    # Compute actual port footprint of each replica: {base, base+2, base+4, base+6} for DP=4.
    occupied = set()
    for p in procs:
        if p.node_rank == 0:  # only the leader-rank entry, but all share base
            dp_size = 4 if p.endpoint_mode == "prefill" else 8
            for r in range(dp_size):
                actual = p.nixl_port + 2 * r
                assert actual not in occupied, (
                    f"port {actual} (replica {p.endpoint_index} {p.endpoint_mode} dp={r}) "
                    f"collides with another replica"
                )
                occupied.add(actual)


# ─────────────────────────────────────────────────────────────────
# TP-only multi-replica NIXL port block (no DP).
# Same root cause as the DP+EP path: vLLM binds base + stride*tp_rank.
# Without per-replica block reservation, 3+ TP replicas/node would collide.
# ─────────────────────────────────────────────────────────────────


def _backend_tp(tp_size_p: int, tp_size_d: int):
    from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
    return VLLMProtocol(
        connector="nixl",
        vllm_config=VLLMServerConfig(
            prefill={"tensor-parallel-size": tp_size_p},
            decode={"tensor-parallel-size": tp_size_d},
        ),
    )


def test_tp_only_nixl_port_same_within_replica():
    """All node-ranks of one TP replica share the same nixl_port base."""
    eps = allocate_endpoints(
        num_prefill=2, num_decode=0, num_agg=0,
        gpus_per_prefill=4, gpus_per_decode=8, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001",),
    )
    procs = _backend_tp(4, 4).endpoints_to_processes(eps)
    by_replica: dict[int, set[int]] = {}
    for p in procs:
        by_replica.setdefault(p.endpoint_index, set()).add(p.nixl_port)
    for idx, ports in by_replica.items():
        assert len(ports) == 1, f"TP replica {idx} has multiple nixl_port values: {ports}"


def test_tp_only_nixl_port_gap_between_replicas_tp4():
    """TP=4 replicas need >=8 port gap (tp_size*2) so vLLM's stride=2 fits."""
    eps = allocate_endpoints(
        num_prefill=4, num_decode=0, num_agg=0,
        gpus_per_prefill=4, gpus_per_decode=8, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001", "bia0002"),
    )
    procs = _backend_tp(4, 4).endpoints_to_processes(eps)
    bases = sorted({p.nixl_port for p in procs})
    assert len(bases) == 4, f"expected 4 distinct TP replica bases, got {bases}"
    gaps = [b - a for a, b in zip(bases, bases[1:])]
    assert all(g >= 8 for g in gaps), f"TP=4 replica-to-replica gap < 8: gaps={gaps}"


def test_tp_only_three_replicas_per_node_no_stride2_overlap():
    """Pre-patch failure case: 3 TP=4 replicas on one node would collide.

    Without the port-block reservation, bases were sequential (e.g. 5400,
    5401, 5402). vLLM TP rank-r binds base + 2*r, so:
      R0 ranks: 5400, 5402, 5404, 5406
      R2 ranks: 5402, 5404, 5406, 5408  ← collides with R0
    With the patch, each replica gets a tp_size*2 = 8 port block, so bases
    are at least 8 apart and no two replicas' stride-2 footprints overlap.
    """
    eps = allocate_endpoints(
        num_prefill=3, num_decode=0, num_agg=0,
        gpus_per_prefill=4, gpus_per_decode=8, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001", "bia0002"),  # 12 GPU total; 3 replicas × TP=4
    )
    procs = _backend_tp(4, 4).endpoints_to_processes(eps)
    occupied: set[int] = set()
    for p in procs:
        if p.node_rank != 0:
            continue  # leader-rank entry carries the replica base
        for r in range(4):  # TP=4 stride-2 footprint
            actual = p.nixl_port + 2 * r
            assert actual not in occupied, (
                f"TP-only port {actual} (replica {p.endpoint_index} tp_rank={r}) "
                f"collides with another replica"
            )
            occupied.add(actual)


def test_tp_only_2p1d_tp4_no_collision():
    """Production reproducer: 2p1d-tp4-tp4 (2 prefill TP=4 + 1 decode TP=4)."""
    eps = allocate_endpoints(
        num_prefill=2, num_decode=1, num_agg=0,
        gpus_per_prefill=4, gpus_per_decode=4, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("bia0001", "bia0002"),
    )
    procs = _backend_tp(4, 4).endpoints_to_processes(eps)
    occupied: set[int] = set()
    for p in procs:
        if p.node_rank != 0:
            continue
        tp = 4
        for r in range(tp):
            actual = p.nixl_port + 2 * r
            assert actual not in occupied, (
                f"2p1d-tp4-tp4 port {actual} (replica {p.endpoint_index} {p.endpoint_mode} "
                f"tp_rank={r}) collides"
            )
            occupied.add(actual)
