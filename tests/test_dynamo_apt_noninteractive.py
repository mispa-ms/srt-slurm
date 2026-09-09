"""The dynamo source build must not stop at a dpkg conffile prompt.

Regression test for a failure that cost a full cluster round-trip to find. On
aws-pdx every cold-cache build exited 100 -- apt-get's own error code -- while
the chain redirected to /dev/null, so nothing said why. Reproduced in the same
container:

    *** ssh_config (Y/I/N/O/D/Z) [default=N] ? dpkg: error processing package
    openssh-client (--configure): ...
    E: Sub-process /usr/bin/dpkg returned an error code (1)

`git` pulls in openssh-client, whose ssh_config collides with the image's copy.
`-y` does not answer conffile prompts.
"""

import shutil
import subprocess

import pytest

from srtctl.core.schema import (
    _apt_install,
    _hash_cached_source_install,
    _live_source_install_for_top_of_tree,
)

DYNAMO_HASH = "ba83080ecd31c1ce918559e576d3c5bc9e092ff1"


@pytest.mark.parametrize(
    "flag",
    [
        "DEBIAN_FRONTEND=noninteractive",
        "-o Dpkg::Options::=--force-confdef",
        "-o Dpkg::Options::=--force-confold",
    ],
)
def test_apt_install_is_noninteractive(flag: str):
    assert flag in _apt_install("git")


def test_apt_install_reports_the_failure_it_hides():
    """Suppressing output is fine; suppressing the *reason* is what cost a day."""
    cmd = _apt_install("git")
    assert "> /dev/null" not in cmd, "a silent apt failure is undiagnosable"
    assert "tail -40" in cmd


@pytest.mark.parametrize(
    "builder",
    [
        lambda: _hash_cached_source_install(DYNAMO_HASH),
        _live_source_install_for_top_of_tree,
    ],
)
def test_every_install_path_is_noninteractive(builder):
    """Both the cached and the live-HEAD recipes install build tools."""
    cmd = builder()
    assert "apt-get install" in cmd
    # No BARE `apt-get install` may survive anywhere in the chain: every one of
    # them has to carry the full non-interactive prefix. Counting --force-confold
    # would not show this, since the prefix is also applied to `apt-get update`.
    guarded = (
        "DEBIAN_FRONTEND=noninteractive apt-get "
        "-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold "
        "install"
    )
    assert cmd.count("apt-get install") == cmd.count(guarded)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    "builder",
    [
        lambda: _hash_cached_source_install(DYNAMO_HASH),
        _live_source_install_for_top_of_tree,
        lambda: _apt_install("git curl"),
    ],
)
def test_rendered_bash_parses(builder):
    """A quoting slip here fails an hour into a cluster job, not at import."""
    cmd = builder()
    probe = f"{cmd} true" if cmd.rstrip().endswith("&&") else cmd
    result = subprocess.run(
        ["bash", "-n"], input=probe, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
