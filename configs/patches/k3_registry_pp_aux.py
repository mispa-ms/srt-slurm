#!/usr/bin/env python3
"""Add supports_pp_aux_hidden_states plumbing to vLLM's model registry.

Two edits that a context diff cannot survive:

  1. a new ``_ModelInfo`` dataclass field. It carries a default, so it MUST be
     the last field — anything else is a "non-default argument follows default
     argument" TypeError at import. A diff anchors on whichever fields happen to
     surround it, and that list drifts between vLLM builds.
  2. a new kwarg in ``_ModelInfo.from_model_cls``'s constructor call. Same
     problem: the diff anchors on neighbouring kwargs.

Pipeline 60754095 lost both hunks that way while every other file applied at an
offset. Anchoring on the class/function structure instead is drift-proof.

Idempotent: re-running is a no-op. Exits non-zero on anything unexpected rather
than writing a half-edited registry.
"""

import re
import sys

FIELD = "    supports_pp_aux_hidden_states: bool = False\n"
KWARG = (
    "            supports_pp_aux_hidden_states=getattr(\n"
    '                model, "supports_pp_aux_hidden_states", False\n'
    "            ),\n"
)
MARKER = "supports_pp_aux_hidden_states"
ACCESSOR = """
    def model_supports_pp_aux_hidden_states(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_pp_aux_hidden_states
"""


def insert_field(lines: list[str]) -> list[str]:
    """Append the field as the LAST field of the _ModelInfo dataclass body."""
    start = next(
        (i for i, l in enumerate(lines) if l.startswith("class _ModelInfo:")), None
    )
    if start is None:
        sys.exit("ERROR: class _ModelInfo not found")

    # The dataclass body runs until its first method (decorator or def).
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if re.match(r"    (@|def )", lines[i])
        ),
        None,
    )
    if end is None:
        sys.exit("ERROR: no method found after class _ModelInfo")

    # Step back over blank lines so the field lands flush with the last field.
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1

    if not re.match(r"    \w+: ", lines[insert_at - 1]):
        sys.exit(f"ERROR: expected a field before line {insert_at}, got {lines[insert_at - 1]!r}")

    return lines[:insert_at] + [FIELD] + lines[insert_at:]


def insert_kwarg(lines: list[str]) -> list[str]:
    """Add the kwarg right after `return _ModelInfo(`. Kwarg order is free."""
    idx = next(
        (i for i, l in enumerate(lines) if l.strip() == "return _ModelInfo("), None
    )
    if idx is None:
        sys.exit("ERROR: `return _ModelInfo(` not found")
    return lines[: idx + 1] + [KWARG] + lines[idx + 1 :]


def insert_accessor(lines: list[str]) -> list[str]:
    """Add the registry lookup right after is_pp_supported_model.

    Its diff hunk only matched with fuzz 2 in pipeline 60754095 — one more build
    of drift and it would have failed like the other two.
    """
    start = next(
        (i for i, l in enumerate(lines) if l.strip().startswith("def is_pp_supported_model(")),
        None,
    )
    if start is None:
        sys.exit("ERROR: def is_pp_supported_model not found")
    end = next(
        (i for i in range(start, len(lines)) if lines[i].strip().startswith("return ")),
        None,
    )
    if end is None:
        sys.exit("ERROR: no return in is_pp_supported_model")
    return lines[: end + 1] + [ACCESSOR] + lines[end + 1 :]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: k3_registry_pp_aux.py <path to vllm/model_executor/models/registry.py>")
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print("registry.py already carries supports_pp_aux_hidden_states")
        return

    lines = src.splitlines(keepends=True)
    lines = insert_field(lines)
    lines = insert_kwarg(lines)
    lines = insert_accessor(lines)
    out = "".join(lines)

    # Both edits or neither: a registry with the field but not the getattr would
    # report False for every model and silently keep PP rejected.
    if FIELD not in out:
        sys.exit("ERROR: _ModelInfo field missing after edit")
    if KWARG not in out:
        sys.exit("ERROR: from_model_cls kwarg missing after edit")
    if "def model_supports_pp_aux_hidden_states(" not in out:
        sys.exit("ERROR: registry accessor missing after edit")
    compile(out, path, "exec")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("registry.py patched: field + from_model_cls kwarg + accessor")


if __name__ == "__main__":
    main()
