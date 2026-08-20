# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Process registry for managing and monitoring spawned processes.

This module provides lifecycle management for srun processes, including:
- Process registration and tracking
- Health monitoring via background thread
- Graceful cleanup on exit or failure
"""

import logging
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TerminationOutcome:
    """How local process termination completed."""

    reaped: bool
    force_killed: bool


def terminate_and_reap(
    popen: subprocess.Popen, *, terminate_timeout: float = 10.0, kill_timeout: float = 5.0
) -> TerminationOutcome:
    """Terminate, then kill, while preserving whether SIGKILL was required."""
    if popen.poll() is not None:
        return TerminationOutcome(reaped=True, force_killed=False)
    popen.terminate()
    try:
        popen.wait(timeout=terminate_timeout)
        return TerminationOutcome(reaped=True, force_killed=False)
    except subprocess.TimeoutExpired:
        logger.warning("Process did not terminate, killing...")
    popen.kill()
    try:
        popen.wait(timeout=kill_timeout)
        return TerminationOutcome(reaped=True, force_killed=True)
    except subprocess.TimeoutExpired:
        logger.error("Process was not reaped after SIGKILL")
        return TerminationOutcome(reaped=False, force_killed=True)


@dataclass
class ManagedProcess:
    """A process managed by the registry.

    Attributes:
        name: Human-readable process name (e.g., "prefill_0", "decode_1")
        popen: The subprocess.Popen object
        log_file: Path to the process log file
        node: Node hostname where the process runs
        critical: If True, failure triggers full cleanup
    """

    name: str
    popen: subprocess.Popen
    log_file: Path | None = None
    node: str | None = None
    critical: bool = True

    @property
    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.popen.poll() is None

    @property
    def exit_code(self) -> int | None:
        """Get exit code if process has exited, None otherwise."""
        return self.popen.poll()

    def terminate(self, timeout: float = 10.0) -> None:
        """Terminate the process gracefully, then kill if needed."""
        if not self.is_running:
            return

        outcome = terminate_and_reap(self.popen, terminate_timeout=timeout, kill_timeout=5)
        if not outcome.reaped:
            logger.error("Process %s was not reaped after SIGKILL", self.name)


# Type alias for named process collections
NamedProcesses = dict[str, ManagedProcess]


class ProcessRegistry:
    """Registry for managing multiple processes with health monitoring.

    Features:
    - Tracks all spawned processes by name
    - Background thread monitors for unexpected exits
    - Graceful cleanup on signal or failure
    - Detailed failure reporting with log tails

    Usage:
        registry = ProcessRegistry(job_id="12345")
        registry.add_process(managed_proc)
        # ... run workload ...
        if registry.check_failures():
            registry.cleanup()
    """

    def __init__(self, job_id: str):
        """Initialize the registry.

        Args:
            job_id: SLURM job ID for logging
        """
        self.job_id = job_id
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()
        self._failed_processes: list[str] = []

    def add_process(self, process: ManagedProcess) -> None:
        """Add a process to the registry.

        Args:
            process: ManagedProcess to track
        """
        with self._lock:
            if process.name in self._processes:
                logger.warning("Replacing existing process '%s' in registry", process.name)
            self._processes[process.name] = process
            logger.debug("Registered process: %s (pid=%d)", process.name, process.popen.pid)

    def add_processes(self, processes: NamedProcesses) -> None:
        """Add multiple processes to the registry.

        Args:
            processes: Dict mapping names to ManagedProcess objects
        """
        for name, proc in processes.items():
            # Ensure the name matches
            if proc.name != name:
                proc = ManagedProcess(
                    name=name,
                    popen=proc.popen,
                    log_file=proc.log_file,
                    node=proc.node,
                    critical=proc.critical,
                )
            self.add_process(proc)

    def check_failures(self) -> bool:
        """Check if any critical process has failed.

        Returns:
            True if any critical process has exited with non-zero code
        """
        with self._lock:
            for name, proc in self._processes.items():
                if proc.critical and not proc.is_running:
                    exit_code = proc.exit_code
                    if exit_code != 0 and name not in self._failed_processes:
                        self._failed_processes.append(name)
                        logger.error(
                            "Critical process '%s' exited with code %d",
                            name,
                            exit_code,
                        )

            return len(self._failed_processes) > 0

    def cleanup(self) -> None:
        """Terminate all registered processes."""
        with self._lock:
            logger.info("Cleaning up %d processes...", len(self._processes))
            for name, proc in self._processes.items():
                if proc.is_running:
                    logger.debug("Terminating process: %s", name)
                    try:
                        proc.terminate()
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to terminate %s: %s", name, e)

    #: Lines of a dead worker's log to quote. A vLLM engine-core startup failure
    #: re-raises through ~40 stack frames, so the old default of 50 was entirely
    #: traceback and the line naming the cause -- which *precedes* the traceback --
    #: never reached the CI log. It hid four separate failures on the GB300
    #: wide-EP arms before anyone read the number.
    #:
    #: This is a constant rather than an env var on purpose: the reader is the
    #: orchestrator process, and neither `environment:` nor `*_environment:` in a
    #: recipe reaches it -- both are rendered into the workers' srun. There was no
    #: way to raise it from a config, which is how the first fix silently did
    #: nothing.
    DEFAULT_FAILURE_TAIL_LINES = 200

    #: Lines matching this, and not looking like a traceback frame, are the
    #: candidates for "See root cause above".
    _CAUSE_RE = re.compile(
        r"(Error|Exception|Traceback|Assertion|assert |FATAL|CRITICAL|"
        r"NotImplemented|Unsupported|not supported|refus|Killed|out of memory|OOM)",
        re.IGNORECASE,
    )
    _FRAME_RE = re.compile(r'^\s*(File "|\^+\s*$|\w+\(.*\)\s*$)')

    @classmethod
    def _likely_causes(cls, lines: list[str], limit: int = 12) -> list[str]:
        """Lines that look like a stated cause rather than a stack frame."""
        out = [
            ln
            for ln in lines
            if cls._CAUSE_RE.search(ln) and not cls._FRAME_RE.match(ln)
        ]
        return out[-limit:]

    def print_failure_details(self, tail_lines: int | None = None) -> None:
        """Print detailed failure information including log tails.

        Prints two things per dead process: the lines that look like a stated
        cause, and then the tail. The cause extract exists because vLLM's own
        message says "See root cause above" and the tail alone routinely does not
        reach far enough above.

        Args:
            tail_lines: Lines to show from each failed process log.
        """
        if tail_lines is None:
            tail_lines = self.DEFAULT_FAILURE_TAIL_LINES
        if not self._failed_processes:
            return

        logger.error("=" * 60)
        logger.error("FAILURE DETAILS")
        logger.error("=" * 60)

        with self._lock:
            for name in self._failed_processes:
                proc = self._processes.get(name)
                if not proc:
                    continue

                logger.error("\n--- Process: %s ---", name)
                logger.error("Exit code: %s", proc.exit_code)
                logger.error("Node: %s", proc.node or "unknown")
                logger.error("Log file: %s", proc.log_file or "none")

                # Tail the log file if available
                if proc.log_file and proc.log_file.exists():
                    try:
                        lines = proc.log_file.read_text().splitlines()
                        if lines:
                            causes = self._likely_causes(lines)
                            if causes:
                                logger.error("\nLikely cause(s), newest last:")
                                for line in causes:
                                    logger.error("  %s", line)
                            logger.error("\nLast %d lines of log:", tail_lines)
                            for line in lines[-tail_lines:]:
                                logger.error("  %s", line)
                    except Exception as e:  # noqa: BLE001
                        logger.error("Could not read log file: %s", e)

        logger.error("=" * 60)

    def get_process(self, name: str) -> ManagedProcess | None:
        """Get a process by name."""
        with self._lock:
            return self._processes.get(name)

    def get_all_processes(self) -> dict[str, ManagedProcess]:
        """Get a copy of all registered processes."""
        with self._lock:
            return dict(self._processes)

    @property
    def process_count(self) -> int:
        """Get the number of registered processes."""
        with self._lock:
            return len(self._processes)


def setup_signal_handlers(
    stop_event: threading.Event,
    registry: ProcessRegistry,
) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        stop_event: Event to signal shutdown
        registry: ProcessRegistry to cleanup on signal
    """

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning("Received signal %s, initiating cleanup...", sig_name)
        stop_event.set()
        registry.cleanup()
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def start_process_monitor(
    stop_event: threading.Event,
    registry: ProcessRegistry,
    poll_interval: float = 2.0,
) -> threading.Thread:
    """Start a background thread that monitors for process failures.

    Args:
        stop_event: Event that signals the monitor to stop
        registry: ProcessRegistry to monitor
        poll_interval: Seconds between checks

    Returns:
        The monitoring thread (already started)
    """

    def monitor_loop():
        while not stop_event.is_set():
            if registry.check_failures():
                logger.error("Critical process failure detected!")
                stop_event.set()
                registry.cleanup()
                sys.exit(1)
            time.sleep(poll_interval)

    thread = threading.Thread(
        target=monitor_loop,
        daemon=True,
        name="ProcessMonitor",
    )
    thread.start()
    return thread
