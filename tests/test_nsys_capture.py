# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM nsys capture work track.

Covers the three behavioral changes introduced for DP+EP nsys profiling:

  1. Worker nsys output path includes GPU index (no per-node clobber).
  2. /engine/start_profile fans out to every decode DP rank under dynamo
     (no single-rank profile asymmetry).
  3. The wrapper.step() symmetry patches in vllm-container-deps-nsys.sh
     apply cleanly against the v0.21.0 gpu_worker.py shape.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = REPO_ROOT / "configs" / "patches" / "vllm-container-deps-nsys.sh"


# ---------------------------------------------------------------------------
# 1. Worker nsys output path includes GPU index
# ---------------------------------------------------------------------------


class TestNsysOutputPath:
    """The nsys output prefix must include sorted gpu indices so per-DP-rank
    processes on the same node don't share an output file."""

    @staticmethod
    def _stub_worker_stage(profiling_type: str = "nsys", workers: list[str] | None = None):
        """Build a minimal WorkerStageMixin instance + a mock process that
        exercises only the nsys_output path construction."""
        from srtctl.cli.mixins.worker_stage import WorkerStageMixin

        worker_nodes = workers or ["theia0127", "theia0128", "theia0129"]

        class _Stage(WorkerStageMixin):
            def __init__(self):
                self.config = MagicMock()
                self.config.profiling.enabled = True
                self.config.profiling.is_nsys = profiling_type == "nsys"
                self.config.profiling.is_torch = profiling_type == "torch"
                self.config.profiling.get_nsys_prefix = MagicMock(return_value=["nsys", "profile"])
                self.config.frontend.type = "dynamo"
                self.config.backend_type = "vllm"
                self.config.backend = MagicMock()
                self.config.backend.get_environment_for_mode.return_value = {}
                self.config.backend.build_worker_command.return_value = ["echo", "test"]
                self.config.observability = MagicMock(spec=[])
                # Disable preamble injection (would otherwise try to join
                # MagicMock instances into the bash preamble).
                self.config.dynamo.install = False
                self.config.frontend.setup_script = None
                self.config.backend.setup_script = None
                # KVBM endpoint env defaults
                self.config.kvbm = None
                self.runtime = MagicMock()
                self.runtime.log_dir = Path("/tmp/_nsys_test")
                self.runtime.head_node_ip = "10.0.0.1"
                self.runtime.nodes.infra = "10.0.0.1"
                self.runtime.nodes.worker = worker_nodes
                self.runtime.environment = {}
                # The dummy-fill DP check at start_worker tail compares
                # `len(process.gpu_indices) < self.runtime.gpus_per_node`
                self.runtime.gpus_per_node = 8

        return _Stage()

    @staticmethod
    def _process(gpu_indices, *, node="theia0128", endpoint_mode="decode", endpoint_index=0):
        from srtctl.core.topology import Process

        return Process(
            node=node,
            gpu_indices=frozenset(gpu_indices),
            sys_port=8081,
            http_port=30000,
            endpoint_mode=endpoint_mode,
            endpoint_index=endpoint_index,
            node_rank=0,
        )

    def _run_and_capture_nsys_output(self, gpu_indices, **kwargs):
        from srtctl.cli.mixins import worker_stage as worker_stage_module

        stage = self._stub_worker_stage()
        proc = self._process(gpu_indices, **kwargs)

        with (
            patch.object(worker_stage_module, "start_srun_process", return_value=MagicMock()),
            patch.object(worker_stage_module, "build_otel_env", return_value={}),
            patch.object(Path, "mkdir", return_value=None),
        ):
            stage.start_worker(proc, [])

        # nsys_output is the first positional arg to get_nsys_prefix
        assert stage.config.profiling.get_nsys_prefix.called
        nsys_output = stage.config.profiling.get_nsys_prefix.call_args[0][0]
        return nsys_output

    def test_single_gpu_index(self):
        out = self._run_and_capture_nsys_output([0])
        assert out == "/logs/profiles/decode/theia0128_decode_w0_gpu0_profile"

    def test_multiple_gpu_indices_sorted(self):
        # frozenset preserves no order; sort must come from inside worker_stage.
        out = self._run_and_capture_nsys_output([3, 0, 2, 1])
        assert out == "/logs/profiles/decode/theia0128_decode_w0_gpu0-1-2-3_profile"

    def test_no_gpu_indices_falls_back(self):
        """Empty gpu_indices should produce the legacy path (no _gpu tag)."""
        out = self._run_and_capture_nsys_output([])
        assert out == "/logs/profiles/decode/theia0128_decode_w0_profile"
        assert "_gpu" not in out

    def test_prefill_mode_path(self):
        out = self._run_and_capture_nsys_output([0, 1], endpoint_mode="prefill", node="theia0127")
        assert out == "/logs/profiles/prefill/theia0127_prefill_w0_gpu0-1_profile"


# ---------------------------------------------------------------------------
# 2. /engine/start_profile fans out to every decode DP rank under dynamo
# ---------------------------------------------------------------------------


class TestDecodeEndpointFanout:
    """Under Dynamo + multi-DP decode, PROFILE_DECODE_ENDPOINTS must list
    every DP rank's endpoint. PROFILE_DECODE_IPS stays leader-only."""

    @staticmethod
    def _stub_benchmark_stage(frontend_type: str, processes):
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        # backend_processes / endpoints are @property in the mixin that
        # raise NotImplementedError; override them on the subclass.
        class _Stage(BenchmarkStageMixin):
            def __init__(self):
                self.config = MagicMock()
                self.config.frontend.type = frontend_type
                self.config.served_model_name = "test-model"
                # Profiling: enabled, type=nsys, no per-phase step ranges
                self.config.profiling = MagicMock()
                self.config.profiling.enabled = True
                self.config.profiling.type = "nsys"
                self.config.profiling.is_torch = False
                self.config.profiling.prefill = None
                self.config.profiling.decode = None
                self.config.profiling.aggregated = None
                self.runtime = MagicMock()
                self.runtime.network_interface = "eth0"
                self.runtime.nodes.head = "theia0127"
                self.runtime.frontend_port = 8000

            @property
            def backend_processes(self):
                return processes

            @property
            def endpoints(self):
                return []

        return _Stage()

    @staticmethod
    def _decode_process(*, node, sys_port, is_leader):
        # Construct a lightweight stand-in (real Process is frozen and has
        # many required fields). The benchmark_stage code only reads:
        #   .endpoint_mode, .is_leader, .node, .sys_port, .http_port
        proc = MagicMock()
        proc.endpoint_mode = "decode"
        proc.is_leader = is_leader
        proc.node = node
        proc.sys_port = sys_port
        proc.http_port = 30000
        return proc

    def _collect_env(self, frontend_type, processes):
        """Invoke the IPs/endpoints loop and return the resulting env dict."""
        from srtctl.cli.mixins import benchmark_stage as bench_module

        stage = self._stub_benchmark_stage(frontend_type, processes)
        runner = MagicMock()
        runner.name = "SA-Bench"

        with patch.object(bench_module, "get_hostname_ip", side_effect=lambda node, _iface: f"10.0.0.{node[-1]}"):
            env = stage._get_benchmark_profiling_env(runner)

        return env

    def test_dynamo_decode_fans_out_to_all_ranks(self):
        processes = [
            self._decode_process(node="theia0128", sys_port=8081, is_leader=True),
            self._decode_process(node="theia0128", sys_port=8082, is_leader=False),
            self._decode_process(node="theia0129", sys_port=8081, is_leader=True),
            self._decode_process(node="theia0129", sys_port=8082, is_leader=False),
        ]
        env = self._collect_env("dynamo", processes)

        endpoints = env.get("PROFILE_DECODE_ENDPOINTS", "").split(",")
        # All 4 ranks must be listed in ENDPOINTS
        assert sorted(endpoints) == sorted(
            [
                "10.0.0.8:8081",
                "10.0.0.8:8082",
                "10.0.0.9:8081",
                "10.0.0.9:8082",
            ]
        )
        # IPS stays leader-only (one per node)
        ips = env.get("PROFILE_DECODE_IPS", "").split(",")
        assert sorted(ips) == ["10.0.0.8", "10.0.0.9"]

    def test_non_dynamo_decode_is_leader_only(self):
        processes = [
            self._decode_process(node="theia0128", sys_port=8081, is_leader=True),
            self._decode_process(node="theia0128", sys_port=8082, is_leader=False),
            self._decode_process(node="theia0129", sys_port=8081, is_leader=True),
            self._decode_process(node="theia0129", sys_port=8082, is_leader=False),
        ]
        env = self._collect_env("sglang", processes)

        # Non-dynamo frontends (sglang router etc.) emit only the leader
        endpoints = env.get("PROFILE_DECODE_ENDPOINTS", "").split(",")
        # http_port is used for non-dynamo; both leaders share http_port=30000
        assert endpoints == ["10.0.0.8:30000", "10.0.0.9:30000"]


# ---------------------------------------------------------------------------
# 3. wrapper.step() AST patches apply cleanly to the v0.21.0 gpu_worker shape
# ---------------------------------------------------------------------------


_STUB_GPU_WORKER = """\
from contextlib import nullcontext


class Worker:
    def __init__(self):
        self.model_runner = None
        self.profiler = None

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:
            return nullcontext()

        self.profiler.step()
        # ... rest of annotate_profile body would follow ...
        return nullcontext()
"""


class TestWrapperStepPatches:
    """The two python heredocs inside vllm-container-deps-nsys.sh must
    successfully rewrite a stub gpu_worker.py with the v0.21.0 block
    shape, and must be idempotent (re-running is a no-op)."""

    @staticmethod
    def _extract_heredoc(name: str) -> str:
        src = PATCH_SCRIPT.read_text()
        m = re.search(rf"python3 - <<'{name}'\n(.*?)\n{name}\n", src, re.DOTALL)
        assert m is not None, f"heredoc {name} not found in {PATCH_SCRIPT}"
        return m.group(1)

    @staticmethod
    def _materialize_stub_gpu_worker(tmp_path: Path) -> Path:
        # The python heredocs check two hardcoded candidate paths:
        #   /usr/local/lib/python3.{12,10}/dist-packages/vllm/v1/worker/gpu_worker.py
        # The test can't write to /usr/local, but the heredoc body is
        # self-contained — we modify it inline to point at a tmp path
        # instead of patching imports.
        target = tmp_path / "gpu_worker.py"
        target.write_text(_STUB_GPU_WORKER)
        return target

    def _run_heredoc_against(self, heredoc_body: str, stub_path: Path) -> subprocess.CompletedProcess:
        # Swap the candidate path list to point at the tmp stub so the test
        # is hermetic.
        rewritten = heredoc_body.replace(
            'pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")',
            f'pathlib.Path("{stub_path}")',
        ).replace(
            'pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_worker.py")',
            f'pathlib.Path("{stub_path}.unused")',
        )
        return subprocess.run(
            [sys.executable, "-c", rewritten],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_dummy_tick_applies(self, tmp_path):
        stub = self._materialize_stub_gpu_worker(tmp_path)
        body = self._extract_heredoc("PYPATCH_DUMMY_TICK")
        result = self._run_heredoc_against(body, stub)
        assert result.returncode == 0, result.stderr
        new = stub.read_text()
        assert "self.profiler.step()" in new
        # The tick must precede the _dummy_run call
        tick_idx = new.find("self.profiler.step()")
        dummy_idx = new.find("self.model_runner._dummy_run")
        assert tick_idx < dummy_idx
        # And specifically inside execute_dummy_batch
        edb_idx = new.find("def execute_dummy_batch")
        ap_idx = new.find("def annotate_profile")
        assert edb_idx < tick_idx < ap_idx

    def test_dummy_tick_idempotent(self, tmp_path):
        stub = self._materialize_stub_gpu_worker(tmp_path)
        body = self._extract_heredoc("PYPATCH_DUMMY_TICK")
        r1 = self._run_heredoc_against(body, stub)
        first = stub.read_text()
        r2 = self._run_heredoc_against(body, stub)
        second = stub.read_text()
        assert r1.returncode == 0 and r2.returncode == 0
        assert first == second
        assert "Already patched, skipping" in r2.stdout

    def test_annotate_gate_applies(self, tmp_path):
        stub = self._materialize_stub_gpu_worker(tmp_path)
        body = self._extract_heredoc("PYPATCH_ANNOTATE_GATE")
        result = self._run_heredoc_against(body, stub)
        assert result.returncode == 0, result.stderr
        new = stub.read_text()
        # Tick is now gated on tokens > 0
        assert "if scheduler_output.total_num_scheduled_tokens > 0:" in new
        # And the unconditional tick (4 spaces of indent) is gone
        assert "\n        self.profiler.step()\n" not in new

    def test_annotate_gate_idempotent(self, tmp_path):
        stub = self._materialize_stub_gpu_worker(tmp_path)
        body = self._extract_heredoc("PYPATCH_ANNOTATE_GATE")
        r1 = self._run_heredoc_against(body, stub)
        first = stub.read_text()
        r2 = self._run_heredoc_against(body, stub)
        second = stub.read_text()
        assert r1.returncode == 0 and r2.returncode == 0
        assert first == second
        assert "Already patched, skipping" in r2.stdout

    def test_both_patches_compose(self, tmp_path):
        """Running DUMMY_TICK then ANNOTATE_GATE on the same file produces
        the canonical symmetric counter shape: dummy ticks always, annotate
        ticks gated."""
        stub = self._materialize_stub_gpu_worker(tmp_path)
        for name in ("PYPATCH_DUMMY_TICK", "PYPATCH_ANNOTATE_GATE"):
            body = self._extract_heredoc(name)
            result = self._run_heredoc_against(body, stub)
            assert result.returncode == 0, f"{name} failed: {result.stderr}"
        new = stub.read_text()
        # Must contain BOTH patched blocks
        assert "if self.profiler is not None:" in new
        assert "if scheduler_output.total_num_scheduled_tokens > 0:" in new
        # And must still be valid python
        compile(new, str(stub), "exec")

    def test_fails_loud_on_source_drift(self, tmp_path):
        """If upstream vLLM changes the block shape, the patch must fail
        hard (non-zero exit) rather than silently no-op. A silent skip
        would let the asymmetric-tick cascade quietly re-emerge on a
        newer vLLM image."""
        original = "# vLLM source has drifted; no execute_dummy_batch in this shape\n"
        for name in ("PYPATCH_DUMMY_TICK", "PYPATCH_ANNOTATE_GATE"):
            target = tmp_path / f"gpu_worker_{name}.py"
            target.write_text(original)
            body = self._extract_heredoc(name)
            result = self._run_heredoc_against(body, target)
            assert result.returncode != 0, f"{name} should hard-fail on drift, got {result.returncode}"
            assert "ERROR" in result.stdout
            assert "drifted" in result.stdout
            # File must be untouched
            assert target.read_text() == original

    def test_install_script_parses_with_bash(self):
        """Defense-in-depth: the install script must be syntactically valid bash."""
        result = subprocess.run(["bash", "-n", str(PATCH_SCRIPT)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
