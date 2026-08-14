"""Print the root cause last, so it survives the 50-line window.

srt-slurm quotes only the final 50 lines of a failed process log. A vLLM engine
failure ends with asyncio, uvloop and EngineCore frames -- roughly forty lines of
"raise RuntimeError(Engine core initialization failed. See root cause above)" --
so the exception that actually explains it is pushed out of view every time. Four
arms died at decode engine-core init today and the trace showed only the wrapper.

This installs an excepthook that walks the __cause__/__context__ chain to the
innermost exception and prints it as the last thing the process writes. Nothing
else changes: the original traceback is printed first, exactly as before.

Installed as sitecustomize.py, which CPython imports automatically at startup, so
it covers every worker without touching a launch command.
"""

import sys


def _install() -> None:
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        previous(exc_type, exc, tb)
        try:
            root = exc
            seen = set()
            # __cause__ first ("raise X from Y"), then __context__ for implicit
            # chaining; guard against cycles, which a chained re-raise can make.
            while True:
                nxt = root.__cause__ or root.__context__
                if nxt is None or id(nxt) in seen:
                    break
                seen.add(id(root))
                root = nxt
            msg = str(root).strip().splitlines()
            head = msg[0] if msg else ""
            sys.stderr.write(
                f"\n[k3] ROOT CAUSE: {type(root).__name__}: {head[:400]}\n"
            )
            if root is not exc:
                sys.stderr.write(
                    f"[k3]   surfaced as: {type(exc).__name__}: "
                    f"{str(exc).strip().splitlines()[0][:200]}\n"
                )
            sys.stderr.flush()
        except Exception:
            # A diagnostic must never become the failure it is reporting.
            pass

    sys.excepthook = hook


_install()
