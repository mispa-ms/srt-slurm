# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM NIXL side-channel port stride-aware allocation.

vLLM 0.22.x binds the NIXL handshake listener at:
    actual_port = VLLM_NIXL_SIDE_CHANNEL_PORT + stride * rank
with stride=2 (was 1 in 0.21.x). The per-replica NIXL block must therefore
be `parallel_size * stride` ports wide, not `parallel_size`, to avoid
adjacent replicas colliding on the strided ports of the previous block.

These tests cover both code paths:
  * DP-EP mode (data-parallel-size set) — block sized by dp_size * stride
  * TP-only mode (no data-parallel-size)  — block sized by tp_size * stride
"""

from __future__ import annotations

from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
from srtctl.core.topology import Endpoint, NodePortAllocator
from srtctl.ports import VLLM_NIXL_PORT_BASE, VLLM_NIXL_PORT_STRIDE_DEFAULT


def _make_backend(prefill: dict | None = None, decode: dict | None = None, stride: int | None = None) -> VLLMProtocol:
    cfg = VLLMServerConfig(prefill=prefill, decode=decode)
    kwargs: dict = {"vllm_config": cfg}
    if stride is not None:
        kwargs["nixl_port_stride"] = stride
    return VLLMProtocol(**kwargs)


class TestNixlStrideDPMode:
    """DP-EP mode: nixl block per replica = dp_size * nixl_port_stride."""

    def test_single_dp_replica_reserves_dp_size_times_stride(self):
        backend = _make_backend(prefill={"data-parallel-size": 4})
        endpoints = [
            Endpoint(mode="prefill", index=0, nodes=("node0",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
        ]
        allocator = NodePortAllocator()
        backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        # 1 replica × dp=4 × stride=2 = 8 ports consumed
        expected_next = VLLM_NIXL_PORT_BASE + 4 * VLLM_NIXL_PORT_STRIDE_DEFAULT
        assert allocator._next_nixl_port == expected_next

    def test_two_dp_replicas_no_collision(self):
        """Two replicas of dp=4 each must use disjoint port windows.

        With the bug (reserve dp_size only, stride=2 ignored):
            R0 reserves [base..base+3]; R0's binds = base + 2*[0..3] = [base, base+2, base+4, base+6]
            R1 reserves [base+4..base+7]; R1's binds at base+4 → collides with R0's tp_rank=2.
        With the fix (reserve dp_size * stride):
            R0 reserves [base..base+7]; R1 reserves [base+8..base+15]; no overlap.
        """
        backend = _make_backend(prefill={"data-parallel-size": 4})
        # Two prefill endpoints (= two replicas) on separate nodes
        endpoints = [
            Endpoint(mode="prefill", index=0, nodes=("node0",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
            Endpoint(mode="prefill", index=1, nodes=("node1",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)

        # Group processes by endpoint_index → each endpoint's processes share one nixl_port base
        bases_by_endpoint: dict[int, set[int]] = {}
        for p in processes:
            bases_by_endpoint.setdefault(p.endpoint_index, set()).add(p.nixl_port)
        # Each endpoint has a single nixl_port base value
        assert all(len(s) == 1 for s in bases_by_endpoint.values())
        base_r0 = next(iter(bases_by_endpoint[0]))
        base_r1 = next(iter(bases_by_endpoint[1]))
        # R1's base must be at least R0's base + dp_size*stride to fully clear R0's strided range
        assert base_r1 - base_r0 >= 4 * VLLM_NIXL_PORT_STRIDE_DEFAULT, (
            f"adjacent DP replicas collide: R0 base={base_r0}, R1 base={base_r1}, "
            f"need spacing >= {4 * VLLM_NIXL_PORT_STRIDE_DEFAULT}"
        )

    def test_stride_one_legacy_behavior(self):
        """With nixl_port_stride=1 (vLLM 0.21.x), block size = dp_size."""
        backend = _make_backend(prefill={"data-parallel-size": 4}, stride=1)
        endpoints = [
            Endpoint(mode="prefill", index=0, nodes=("node0",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
            Endpoint(mode="prefill", index=1, nodes=("node1",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        bases = sorted({p.nixl_port for p in processes})
        assert bases[1] - bases[0] == 4  # dp_size=4, stride=1


class TestNixlStrideTPMode:
    """TP-only mode (no data-parallel-size): block per replica = tp_size * stride."""

    def test_single_tp_replica_reserves_tp_size_times_stride(self):
        backend = _make_backend(decode={"tensor-parallel-size": 4})
        endpoints = [
            Endpoint(mode="decode", index=0, nodes=("node0",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
        ]
        allocator = NodePortAllocator()
        backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        expected_next = VLLM_NIXL_PORT_BASE + 4 * VLLM_NIXL_PORT_STRIDE_DEFAULT
        assert allocator._next_nixl_port == expected_next

    def test_three_tp_replicas_no_collision(self):
        """N>=3 TP replicas with stride=2 — the smallest collision case.

        With the bug (single port per replica): R0=base, R1=base+1, R2=base+2.
        vLLM binds R0 ranks at [base, base+2, base+4, base+6]; R2's base
        collides with R0's tp_rank=1 port (base+2). Verified by reserving a
        tp_size*stride block per replica.
        """
        backend = _make_backend(decode={"tensor-parallel-size": 4})
        endpoints = [
            Endpoint(mode="decode", index=i, nodes=(f"node{i}",), gpu_indices=frozenset(range(4)), gpus_per_node=8)
            for i in range(3)
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        # One process per node (TP-only path), three distinct base ports
        bases = sorted(p.nixl_port for p in processes)
        assert len(bases) == 3
        assert bases[1] - bases[0] >= 4 * VLLM_NIXL_PORT_STRIDE_DEFAULT
        assert bases[2] - bases[1] >= 4 * VLLM_NIXL_PORT_STRIDE_DEFAULT

    def test_tp_size_one_collapses_to_single_port(self):
        """tp_size=1 (no tensor parallel) reserves only `stride` ports per replica."""
        backend = _make_backend(decode={"tensor-parallel-size": 1})
        endpoints = [
            Endpoint(mode="decode", index=0, nodes=("node0",), gpu_indices=frozenset({0}), gpus_per_node=8),
            Endpoint(mode="decode", index=1, nodes=("node1",), gpu_indices=frozenset({0}), gpus_per_node=8),
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        bases = sorted(p.nixl_port for p in processes)
        assert bases[1] - bases[0] == 1 * VLLM_NIXL_PORT_STRIDE_DEFAULT

    def test_tp_size_implicit_from_gpu_indices(self):
        """When tensor-parallel-size is NOT set, fall back to len(gpu_indices).

        Recipes that omit `tensor-parallel-size` but place 4 GPUs in one
        endpoint still run TP=4 internally — the allocator must reserve
        len(gpu_indices) * stride ports per replica, not 1 * stride.
        """
        # No `tensor-parallel-size` key — config is empty for decode mode
        backend = _make_backend(decode={})
        endpoints = [
            Endpoint(mode="decode", index=i, nodes=(f"node{i}",), gpu_indices=frozenset(range(4)), gpus_per_node=8)
            for i in range(2)
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        bases = sorted(p.nixl_port for p in processes)
        # Implicit TP=4 from len(gpu_indices=range(4))
        assert bases[1] - bases[0] >= 4 * VLLM_NIXL_PORT_STRIDE_DEFAULT, (
            f"implicit-TP fallback failed: R0={bases[0]}, R1={bases[1]}, "
            f"need spacing >= {4 * VLLM_NIXL_PORT_STRIDE_DEFAULT}"
        )


class TestHetGroupPropagation:
    """Regression: heterogeneous SLURM job het_group must reach every Process.

    The earlier upstream `topology.endpoints_to_processes` helper set
    `het_group=endpoint.het_group` on every emitted Process. When the
    stride fix moved both DP and TP paths into VLLMProtocol's inline loop,
    that propagation must be preserved — otherwise srun calls on
    heterogeneous jobs lose their `--het-group=N` flag (see worker_stage
    + slurm.py:264) and workers land in the wrong het component.
    """

    def test_tp_only_prefill_propagates_het_group(self):
        backend = _make_backend(decode={"tensor-parallel-size": 4})
        endpoints = [
            Endpoint(mode="prefill", index=0, nodes=("p0",),
                     gpu_indices=frozenset(range(4)), gpus_per_node=8, het_group=0),
            Endpoint(mode="decode", index=0, nodes=("d0",),
                     gpu_indices=frozenset(range(4)), gpus_per_node=8, het_group=1),
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        prefill = [p for p in processes if p.endpoint_mode == "prefill"]
        decode = [p for p in processes if p.endpoint_mode == "decode"]
        assert all(p.het_group == 0 for p in prefill), "prefill het_group=0 must propagate"
        assert all(p.het_group == 1 for p in decode), "decode het_group=1 must propagate"

    def test_dp_ep_prefill_propagates_het_group(self):
        backend = _make_backend(prefill={"data-parallel-size": 4})
        endpoints = [
            Endpoint(mode="prefill", index=0, nodes=("p0",),
                     gpu_indices=frozenset(range(4)), gpus_per_node=8, het_group=0),
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        assert processes  # sanity: DP creates 1 process per GPU
        assert all(p.het_group == 0 for p in processes)

    def test_multi_node_tp_shares_bootstrap_port(self):
        """Multi-node TP prefill: every node gets the SAME bootstrap port.

        Matches the upstream topology helper's semantics (the bootstrap
        server lives on the leader node; all worker nodes connect to the
        same port number). A regression where only the leader had a
        bootstrap_port would leave non-leader nodes unable to coordinate.
        """
        backend = _make_backend(decode={"tensor-parallel-size": 16})
        endpoints = [
            Endpoint(mode="prefill", index=0, nodes=("p0", "p1"),
                     gpu_indices=frozenset(range(8)), gpus_per_node=8),
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        bootstrap_ports = {p.bootstrap_port for p in processes}
        # exactly one bootstrap port shared across the multi-node prefill endpoint
        assert len(bootstrap_ports) == 1 and None not in bootstrap_ports, (
            f"multi-node TP prefill must share one bootstrap port, got: {bootstrap_ports}"
        )


class TestNixlStrideMixed:
    """Mixed deployment: DP prefill + TP decode (or vice versa)."""

    def test_dp_prefill_plus_tp_decode_disjoint_blocks(self):
        backend = _make_backend(
            prefill={"data-parallel-size": 4},
            decode={"tensor-parallel-size": 8},
        )
        endpoints = [
            Endpoint(mode="prefill", index=0, nodes=("p0",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
            Endpoint(mode="prefill", index=1, nodes=("p1",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
            Endpoint(mode="decode", index=0, nodes=("d0",), gpu_indices=frozenset(range(8)), gpus_per_node=8),
            Endpoint(mode="decode", index=1, nodes=("d1",), gpu_indices=frozenset(range(8)), gpus_per_node=8),
        ]
        allocator = NodePortAllocator()
        processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator)
        # All NIXL bases must be pairwise spaced by at least parallel_size*stride
        prefill_bases = sorted({p.nixl_port for p in processes if p.endpoint_mode == "prefill"})
        decode_bases = sorted({p.nixl_port for p in processes if p.endpoint_mode == "decode"})
        assert len(prefill_bases) == 2 and len(decode_bases) == 2
        # Prefill: dp=4 stride=2 → spacing ≥ 8
        assert prefill_bases[1] - prefill_bases[0] >= 4 * VLLM_NIXL_PORT_STRIDE_DEFAULT
        # Decode: tp=8 stride=2 → spacing ≥ 16
        assert decode_bases[1] - decode_bases[0] >= 8 * VLLM_NIXL_PORT_STRIDE_DEFAULT
