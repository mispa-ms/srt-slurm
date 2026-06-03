# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for setup_head IP selection logic.

Covers:
  * Method 0: explicit network_interface → `ip addr show dev <iface>`
  * 10.* preference: when scanning candidate IPs (Method 1/2/3 results),
    pick a 10.* mgmt IP before falling back to other private ranges.
  * Behavior when network_interface is unset: Method 0 is skipped, the
    10.* preference still applies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from srtctl.cli.setup_head import get_local_ip


class TestExplicitNetworkInterface:
    """Method 0 — `ip addr show dev <iface>` deterministic NIC pick."""

    def test_explicit_interface_returns_matching_ip(self):
        ip_output = "2: os_p2n1    inet 10.66.34.50/24 scope global os_p2n1\n"
        result = MagicMock(returncode=0, stdout=ip_output, stderr="")
        with patch("subprocess.run", return_value=result):
            assert get_local_ip(network_interface="os_p2n1") == "10.66.34.50"

    def test_explicit_interface_missing_falls_through(self):
        # `ip` returns non-zero (interface not found)
        method0 = MagicMock(returncode=1, stdout="", stderr="Device not found")
        # Method 1 (hostname -I) returns a valid 10.* IP
        method1 = MagicMock(returncode=0, stdout="10.66.34.50 172.20.16.16\n", stderr="")
        with patch("subprocess.run", side_effect=[method0, method1]):
            assert get_local_ip(network_interface="missing_nic") == "10.66.34.50"

    def test_explicit_interface_no_inet_line_falls_through(self):
        # `ip` returns output but no inet line (interface up but no IPv4)
        method0 = MagicMock(returncode=0, stdout="2: os_p2n1    inet6 fe80::1/64\n", stderr="")
        method1 = MagicMock(returncode=0, stdout="10.66.34.50\n", stderr="")
        with patch("subprocess.run", side_effect=[method0, method1]):
            assert get_local_ip(network_interface="os_p2n1") == "10.66.34.50"


class TestMgmtRangePreference:
    """10.* preference: pick mgmt range first regardless of `hostname -I` order."""

    def test_prefers_10_dot_over_172_dot(self):
        # No explicit NIC; hostname -I returns IB IP first, mgmt IP second
        method1 = MagicMock(returncode=0, stdout="172.20.16.16 10.66.34.50\n", stderr="")
        with patch("subprocess.run", return_value=method1):
            assert get_local_ip() == "10.66.34.50"

    def test_returns_172_dot_when_no_10_dot_present(self):
        method1 = MagicMock(returncode=0, stdout="172.20.16.16\n", stderr="")
        with patch("subprocess.run", return_value=method1):
            assert get_local_ip() == "172.20.16.16"

    def test_skips_bad_ips_in_10_dot_preference(self):
        # 10.0.0.1 is bad (loopback-adjacent / not routable); skip to next
        method1 = MagicMock(returncode=0, stdout="127.0.0.1 10.66.34.50\n", stderr="")
        with patch("subprocess.run", return_value=method1):
            assert get_local_ip() == "10.66.34.50"


class TestNoNetworkInterface:
    """When network_interface is None (unset) — Method 0 is skipped entirely."""

    def test_none_skips_method0_and_uses_mgmt_preference(self):
        # Only one `subprocess.run` call expected (Method 1), not two
        method1 = MagicMock(returncode=0, stdout="172.20.16.16 10.66.34.50\n", stderr="")
        with patch("subprocess.run", return_value=method1) as mock_run:
            assert get_local_ip(network_interface=None) == "10.66.34.50"
            # Method 0 (ip addr show dev) NOT invoked
            for call in mock_run.call_args_list:
                args = call.args[0] if call.args else call.kwargs.get("args", [])
                assert not (isinstance(args, list) and len(args) >= 4 and args[3] == "addr"), (
                    f"Method 0 should not run when network_interface=None: {args}"
                )
