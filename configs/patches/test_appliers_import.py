#!/usr/bin/env python3
"""Every applier's output must import and run, not merely compile.

    python3 configs/patches/test_appliers_import.py

This exists because of one line. apply-lookup-align.py added a logger.info() to
coordinator.py, which has no logger -- worker.py defines one, coordinator.py
never needed one. The file compiled perfectly. At the first lookup it raised
NameError inside the connector's process_request thread, so the thread died, the
process lived, and the client sat at

    returned=0/531 | sent=51 | in_flight=51 | errors=0 | elapsed=3330.6s

for fifty-five minutes with idle GPUs until the reaper claimed the job. Four
ladders were lost to it and read first as a reaper problem, then as a mooncake
segment problem, then as a suspected retry loop.

`python -m compileall` cannot catch that: the syntax was never wrong. What
catches it is executing the patched function against a stub and asserting no
NameError. That is what this does, offline, in a second.
"""

import pathlib
import subprocess
import sys
import tempfile
import textwrap

HERE = pathlib.Path(__file__).resolve().parent
VLLM = pathlib.Path("/home/scratch.misunp_gpu/repo/vllm")
NIGHTLY = "3d204dfdaaf09d67d49c7855630ef949754e0f8f"

COORD = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py"
WORKER = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"


def nightly(path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(VLLM), "show", f"{NIGHTLY}:{path}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def stage(paths: list[str]) -> pathlib.Path:
    d = pathlib.Path(tempfile.mkdtemp())
    for p in paths:
        body = nightly(p)
        if body is None:
            return None
        f = d / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    return d


def func_source(src: str, name: str) -> str:
    start = src.index(f"    def {name}")
    end = src.index("\n    def ", start + 10)
    return textwrap.dedent(src[start:end])


def check_lookup_align() -> list[str]:
    """The exact failure: align_lookup_length logs from a module with no logger."""
    d = stage([COORD])
    if d is None:
        return ["cannot read coordinator.py from the nightly -- skipped"]
    r = subprocess.run([sys.executable, str(HERE / "apply-lookup-align.py"), str(d)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return [f"apply-lookup-align.py failed: {r.stderr.strip()[:200]}"]

    src = (d / COORD).read_text()
    problems = []
    # Whatever the patch logs from, that module must define it.
    if "logger." in src and "init_logger" not in src:
        problems.append("coordinator.py logs but never imports a logger")

    ns: dict = {}
    if "init_logger" in src:
        class _Logger:
            def info(self, *a, **k):
                pass
        ns["logger"] = _Logger()
    exec(func_source(src, "align_lookup_length"), ns)

    class Stub:
        lcm_block_size = 12288      # DCP8
        hash_block_size = 128
        enable_partial_hash_hits = True

    try:
        out = ns["align_lookup_length"](Stub(), 122880)
    except NameError as e:
        problems.append(f"align_lookup_length raises NameError at runtime: {e}")
    else:
        if out % Stub.lcm_block_size:
            problems.append(
                f"align_lookup_length returned {out}, not a multiple of "
                f"lcm_block_size -- the store cannot answer it")
    return problems


def check_keysample() -> list[str]:
    """The same trap, in the file that does have a logger -- verify it stays true."""
    d = stage([WORKER])
    if d is None:
        return ["cannot read worker.py from the nightly -- skipped"]
    r = subprocess.run([sys.executable, str(HERE / "apply-keysample-log.py"), str(d)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return [f"apply-keysample-log.py failed: {r.stderr.strip()[:200]}"]
    src = (d / WORKER).read_text()
    if "logger." in src and "logger = " not in src:
        return ["worker.py logs but defines no logger"]
    return []


def main() -> int:
    failures = []
    for name, fn in (("lookup-align", check_lookup_align),
                     ("keysample", check_keysample)):
        problems = fn()
        for p in problems:
            failures.append(f"{name}: {p}")
        print(f"  {name}: {'FAIL' if problems else 'ok'}")
        for p in problems:
            print(f"      {p}")
    if failures:
        return 1
    print("all appliers produce modules that import and run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
