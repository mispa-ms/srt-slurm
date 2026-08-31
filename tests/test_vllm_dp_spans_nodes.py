# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A data-parallel rank whose tp x pp is larger than one node.

vLLM supports this: `nnodes_within_dp` is
`nnodes / (data_parallel_size / data_parallel_size_local)`, it builds a separate
`_INNER_DP_WORLD` group when that is above one, and both the CLI and the executor
branch on `node_rank_within_dp`, which vLLM derives as
`node_rank % nnodes_within_dp`. So srtctl's whole job is to report the global node
rank and the endpoint's node count honestly and let vLLM place each node inside
its rank.

**These tests exist because getting it wrong does not crash.** A node placed in
the wrong DP rank still starts, still serves, and still produces numbers -- wrong
ones. There is no runtime signal, so the mapping is pinned here instead.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends import VLLMProtocol, VLLMServerConfig
from srtctl.core.topology import Endpoint


def _endpoint(nodes: tuple[str, ...], gpus_per_node: int = 8) -> Endpoint:
    return Endpoint(
        mode="agg",
        index=0,
        nodes=nodes,
        gpu_indices=frozenset(range(gpus_per_node)),
        gpus_per_node=gpus_per_node,
    )


def _backend(dp: int, tp: int, pp: int) -> VLLMProtocol:
    return VLLMProtocol(
        vllm_config=VLLMServerConfig(
            aggregated={
                "data-parallel-size": dp,
                "tensor-parallel-size": tp,
                "pipeline-parallel-size": pp,
                "enable-expert-parallel": True,
            }
        ),
        dp_launch_mode="per_node",
    )


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.model_path = Path("/model")
    runtime.is_hf_model = False
    runtime.request_plane = "tcp"
    runtime.frontend_port = 8000
    return runtime


class TestSpanningLayout:
    def test_dp2_tp8_pp2_over_four_nodes(self):
        """Two ranks of sixteen GPUs, each spanning two of the four nodes.

        The pairing is what matters: nodes 0 and 1 are one rank, nodes 2 and 3 are
        the other. dp-start-rank must therefore read 0, 0, 1, 1 -- NOT 0, 1, 2, 3,
        which is what the pre-existing per-node accounting would have produced and
        which would have put four ranks where there are two.
        """
        b = _backend(dp=2, tp=8, pp=2)
        procs = b.endpoints_to_processes(
            [_endpoint(("n0", "n1", "n2", "n3"))], frontend_type="dynamo"
        )
        assert [p.node for p in procs] == ["n0", "n1", "n2", "n3"]
        assert [p.node_rank for p in procs] == [0, 0, 1, 1]

    def test_rank_inside_a_node_is_unchanged(self):
        """The shape that already worked must not move.

        dp8 x tp2 = sixteen GPUs over two nodes: four whole ranks per node, so
        the second node's process starts four ranks after the first.
        """
        b = _backend(dp=8, tp=2, pp=1)
        procs = b.endpoints_to_processes(
            [_endpoint(("n0", "n1"))], frontend_type="dynamo"
        )
        assert [p.node_rank for p in procs] == [0, 4]

    def test_rank_that_does_not_divide_a_node_is_refused(self):
        """tp x pp = 12 on 8-GPU nodes would give a rank part of a node."""
        b = _backend(dp=2, tp=6, pp=2)
        with pytest.raises(ValueError, match="does not divide into"):
            b.endpoints_to_processes(
                [_endpoint(("n0", "n1", "n2"))], frontend_type="dynamo"
            )


class TestSpanningCommand:
    def _cmd(self, b, procs, process):
        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            return b.build_worker_command(
                process, procs, _runtime(), frontend_type="dynamo"
            )

    def test_every_node_gets_its_global_rank_and_the_node_count(self):
        """vLLM computes node_rank_within_dp itself; we owe it the global rank."""
        b = _backend(dp=2, tp=8, pp=2)
        procs = b.endpoints_to_processes(
            [_endpoint(("n0", "n1", "n2", "n3"))], frontend_type="dynamo"
        )
        seen = []
        for p in procs:
            cmd = self._cmd(b, procs, p)
            assert "--nnodes" in cmd, "a spanning rank needs the node count"
            assert cmd[cmd.index("--nnodes") + 1] == "4"
            seen.append(cmd[cmd.index("--node-rank") + 1])
            # One rank per launch when the rank spans nodes, never a fraction.
            assert cmd[cmd.index("--data-parallel-size-local") + 1] == "1"
        assert seen == ["0", "1", "2", "3"]

    def test_start_rank_pairs_the_nodes(self):
        """The two flags together are what place a node: rank 0 gets nodes 0-1."""
        b = _backend(dp=2, tp=8, pp=2)
        procs = b.endpoints_to_processes(
            [_endpoint(("n0", "n1", "n2", "n3"))], frontend_type="dynamo"
        )
        pairs = []
        for p in procs:
            cmd = self._cmd(b, procs, p)
            pairs.append(
                (
                    cmd[cmd.index("--node-rank") + 1],
                    cmd[cmd.index("--data-parallel-start-rank") + 1],
                )
            )
        assert pairs == [("0", "0"), ("1", "0"), ("2", "1"), ("3", "1")]

    def test_non_spanning_shape_emits_no_nnodes(self):
        """Adding these flags where a rank fits in a node would change a
        configuration that already runs, so it must not happen."""
        b = _backend(dp=8, tp=2, pp=1)
        procs = b.endpoints_to_processes(
            [_endpoint(("n0", "n1"))], frontend_type="dynamo"
        )
        cmd = self._cmd(b, procs, procs[0])
        assert "--nnodes" not in cmd
        assert cmd[cmd.index("--data-parallel-size-local") + 1] == "4"
