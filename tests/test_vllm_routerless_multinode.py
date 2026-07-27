# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-node aggregate jobs under the routerless vLLM frontend.

`vllm serve` spans nodes itself via --nnodes/--node-rank: rank 0 serves the
OpenAI API and the other ranks join headless. No router process is involved,
which is what lets multi-node TP/TEP run without Dynamo.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends import VLLMProtocol, VLLMServerConfig
from srtctl.core.topology import Endpoint, Process
from srtctl.ports import FRONTEND_PUBLIC_PORT


def _agg_endpoint(nodes: tuple[str, ...]) -> Endpoint:
    return Endpoint(mode="agg", index=0, nodes=nodes, gpu_indices=frozenset(range(8)), gpus_per_node=8)


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.model_path = Path("/model")
    runtime.is_hf_model = False
    runtime.request_plane = "tcp"
    runtime.frontend_port = FRONTEND_PUBLIC_PORT
    return runtime


class TestRouterlessMultiNodeTopology:
    def test_two_node_agg_yields_one_process_per_node(self):
        """Rank 0 binds the public port; the follower gets no HTTP port."""
        backend = VLLMProtocol()

        processes = backend.endpoints_to_processes([_agg_endpoint(("node0", "node1"))], frontend_type="vllm")

        assert len(processes) == 2
        assert [p.node for p in processes] == ["node0", "node1"]
        assert processes[0].http_port == FRONTEND_PUBLIC_PORT
        assert processes[1].http_port == 0
        # The frontend picks the server by is_leader, so exactly one must qualify.
        assert [p.is_leader for p in processes] == [True, False]

    def test_single_node_agg_is_unchanged(self):
        backend = VLLMProtocol()

        processes = backend.endpoints_to_processes([_agg_endpoint(("node0",))], frontend_type="vllm")

        assert len(processes) == 1
        assert processes[0].http_port == FRONTEND_PUBLIC_PORT
        assert processes[0].is_leader

    def test_data_parallel_layout_is_rejected(self):
        """DP needs a router in front, so it must not silently fall through."""
        backend = VLLMProtocol(vllm_config=VLLMServerConfig(aggregated={"data-parallel-size": 2}))

        with pytest.raises(ValueError, match="does not support data-parallel"):
            backend.endpoints_to_processes([_agg_endpoint(("node0", "node1"))], frontend_type="vllm")


class TestRouterlessMultiNodeCommand:
    def _cmd(self, node: str, node_rank: int) -> list[str]:
        backend = VLLMProtocol(vllm_config=VLLMServerConfig(aggregated={"tensor-parallel-size": 16}))
        leader = Process(
            node="node0",
            gpu_indices=frozenset(range(8)),
            sys_port=8081,
            http_port=FRONTEND_PUBLIC_PORT,
            endpoint_mode="agg",
            endpoint_index=0,
            node_rank=0,
        )
        follower = Process(
            node="node1",
            gpu_indices=frozenset(range(8)),
            sys_port=8082,
            http_port=0,
            endpoint_mode="agg",
            endpoint_index=0,
            node_rank=1,
        )
        process = leader if node_rank == 0 else follower
        assert process.node == node
        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            return backend.build_worker_command(process, [leader, follower], _runtime(), frontend_type="vllm")

    def test_rank0_serves_and_is_not_headless(self):
        cmd = self._cmd("node0", 0)

        assert cmd[cmd.index("--nnodes") + 1] == "2"
        assert cmd[cmd.index("--node-rank") + 1] == "0"
        assert cmd[cmd.index("--master-addr") + 1] == "10.0.0.1"
        assert "--headless" not in cmd
        # Still a direct `vllm serve`, not dynamo.
        assert "serve" in cmd and "dynamo.vllm" not in cmd

    def test_follower_rank_is_headless(self):
        cmd = self._cmd("node1", 1)

        assert cmd[cmd.index("--node-rank") + 1] == "1"
        assert "--headless" in cmd

    def test_multi_node_unsets_inherited_vllm_port(self):
        """An inherited VLLM_PORT seeds vLLM's cross-node MQ allocator and races."""
        cmd = self._cmd("node0", 0)

        assert cmd[:3] == ["env", "-u", "VLLM_PORT"]

    def test_single_node_omits_multi_node_flags(self):
        backend = VLLMProtocol(vllm_config=VLLMServerConfig(aggregated={"tensor-parallel-size": 8}))
        process = Process(
            node="node0",
            gpu_indices=frozenset(range(8)),
            sys_port=8081,
            http_port=FRONTEND_PUBLIC_PORT,
            endpoint_mode="agg",
            endpoint_index=0,
            node_rank=0,
        )

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process, [process], _runtime(), frontend_type="vllm")

        assert "--nnodes" not in cmd
        assert "--node-rank" not in cmd
        assert "--headless" not in cmd
        assert cmd[0] != "env"
