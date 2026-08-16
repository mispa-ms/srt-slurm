# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for data-parallel layouts where a rank owns more than one GPU.

Before this, DP mode assumed one GPU per rank: the endpoint check demanded
`data-parallel-size == total_gpus` and `--data-parallel-size-local` was the raw
local GPU count. That admits `tp1 x dpN` and `tpN x dp1` and nothing between,
which is why every archived Kimi-K3 job is one of those two shapes.

A rank owns `tensor-parallel-size` GPUs, so both conversions divide by it. With
tp unset the divisor is 1 and every pre-existing layout is byte-identical --
the pure-DP cases below pin that.
"""

from pathlib import Path
from unittest.mock import MagicMock

from srtctl.backends import VLLMProtocol
from srtctl.backends.vllm import VLLMServerConfig
from srtctl.core.topology import Endpoint
from srtctl.ports import FRONTEND_PUBLIC_PORT


def _endpoint(mode: str, nodes: tuple[str, ...], gpus_per_node: int = 8) -> Endpoint:
    return Endpoint(
        mode=mode,
        index=0,
        nodes=nodes,
        gpu_indices=frozenset(range(gpus_per_node)),
        gpus_per_node=gpus_per_node,
    )


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.model_path = Path("/model")
    runtime.is_hf_model = False
    runtime.request_plane = "tcp"
    runtime.frontend_port = FRONTEND_PUBLIC_PORT
    return runtime


def _backend(**decode_config) -> VLLMProtocol:
    return VLLMProtocol(
        dp_launch_mode="per_node",
        vllm_config=VLLMServerConfig(decode=decode_config),
    )


def _size_local(cmd: list[str]) -> str:
    return cmd[cmd.index("--data-parallel-size-local") + 1]


class TestRankSizing:
    def test_tp_unset_keeps_one_rank_per_gpu(self):
        """The pure-DP layout every existing config uses. Divisor is 1."""
        backend = _backend(**{"data-parallel-size": 8})

        processes = backend.endpoints_to_processes([_endpoint("decode", ("node0",))])

        assert len(processes) == 1
        cmd = backend.build_worker_command(processes[0], processes, _runtime())
        assert _size_local(cmd) == "8"

    def test_mixed_tp_dp_divides_gpus_between_ranks(self):
        """tp4 x dp2 on one 8-GPU node: two ranks, four GPUs each."""
        backend = _backend(**{"data-parallel-size": 2, "tensor-parallel-size": 4})

        processes = backend.endpoints_to_processes([_endpoint("decode", ("node0",))])

        assert len(processes) == 1
        cmd = backend.build_worker_command(processes[0], processes, _runtime())
        assert _size_local(cmd) == "2"

    def test_mixed_tp_dp_spans_nodes(self):
        """tp4 x dp4 over two nodes: each node holds two of the four ranks."""
        backend = _backend(**{"data-parallel-size": 4, "tensor-parallel-size": 4})

        processes = backend.endpoints_to_processes([_endpoint("decode", ("node0", "node1"))])

        assert [p.node_rank for p in processes] == [0, 2]
        cmd = backend.build_worker_command(processes[0], processes, _runtime())
        assert _size_local(cmd) == "2"


class TestValidation:
    def test_rejects_shape_that_does_not_fill_the_allocation(self):
        """tp4 x dp3 is 12 GPUs; the endpoint holds 8."""
        backend = _backend(**{"data-parallel-size": 3, "tensor-parallel-size": 4})

        try:
            backend.endpoints_to_processes([_endpoint("decode", ("node0",))])
        except ValueError as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("expected a ValueError for a 12-GPU shape on 8 GPUs")

    def test_rejects_pure_dp_mismatch_as_before(self):
        """The pre-existing failure mode still fails, with tp defaulting to 1."""
        backend = _backend(**{"data-parallel-size": 4})

        try:
            backend.endpoints_to_processes([_endpoint("decode", ("node0",))])
        except ValueError as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("expected a ValueError for dp4 on 8 GPUs")

    def test_rejects_a_rank_that_would_span_nodes(self):
        """tp8 x dp2 on 4-GPU nodes: a rank straddles two nodes.

        per_node launching gives each node one process owning its resident
        ranks, so a rank has to fit inside a node. This shape used to floor
        local_dp_size to 0 and surface as "KV-event port block size must be at
        least 1" from the port allocator -- eight GB300 arms died on that
        before the shape itself was named.
        """
        backend = _backend(**{"data-parallel-size": 2, "tensor-parallel-size": 8})
        endpoint = _endpoint("decode", ("node0", "node1", "node2", "node3"), gpus_per_node=4)

        try:
            backend.endpoints_to_processes([endpoint])
        except ValueError as exc:
            assert "would span" in str(exc), exc
            assert "tensor-parallel-size=8" in str(exc), exc
        else:
            raise AssertionError("expected a ValueError for a node-spanning DP rank")
