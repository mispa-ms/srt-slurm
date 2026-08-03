#!/usr/bin/env python3
"""Declare Kimi-K3's opt-in for aux hidden states over pipeline parallelism.

PR #50514 gates the PP aux relay on a class attribute that only Kimi-K3 sets.
The attribute lands in the `KimiLinearModel` class body, whose contents drift
between builds — the class gained `SupportsQuant` and `packed_modules_mapping`
between the PR's base and ours, and the container differs again. The diff hunk
failed that way in pipeline 60921865, so the flag is delivered by anchoring on
the class statement instead.

A plain class attribute has no ordering constraint, so inserting it as the first
line of the class body is always valid.

Idempotent. Exits non-zero on an unexpected file rather than half-editing it.
"""

import sys

ANCHOR = "class KimiLinearModel("
FLAG = (
    "    # Aux taps are carried across pipeline stages in the IntermediateTensors\n"
    "    # payload, so EAGLE3-style drafting works under PP for this model.\n"
    "    supports_aux_hidden_states_over_pp = True\n"
)
MARKER = "supports_aux_hidden_states_over_pp"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: k3_optin_flag.py <path to kimi_k3/nvidia/model.py>")
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print("KimiLinearModel already opts in to aux-over-PP")
        return

    lines = src.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith(ANCHOR)), None)
    if start is None:
        sys.exit(f"ERROR: anchor not found: {ANCHOR}")

    # The class statement may span lines; the body starts after the line that
    # closes it.
    idx = start
    while idx < len(lines) and not lines[idx].rstrip().endswith(":"):
        idx += 1
    if idx >= len(lines):
        sys.exit("ERROR: could not find the end of the class statement")

    out = "".join(lines[: idx + 1] + [FLAG] + lines[idx + 1 :])
    if MARKER not in out:
        sys.exit("ERROR: flag missing after edit")
    compile(out, path, "exec")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("KimiLinearModel patched: opts in to aux-over-PP")


if __name__ == "__main__":
    main()
