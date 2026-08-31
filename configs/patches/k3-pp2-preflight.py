"""Does this image refuse the layout we are about to ask for?

Run by kimi-k3-nightly-v8-pp2.sh after the three patches, before the engine
starts. PCP is refused for Kimi-K3 by a hard assert in the model file, and that
refusal was found only after a submission had already been spent; this checks the
same shape for PP rather than assuming it differs.

MATCH THE STATEMENT, NOT THE TEXT -- this file exists as a separate module
because its first version did not. That version grepped for the string
``pipeline_parallel_size == 1`` and killed six jobs on two hits that are
capability gates, not refusals:

    self.use_sequence_parallel = (
        parallel_config.pipeline_parallel_size == 1
        and parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and (use_mega_moe or parallel_config.data_parallel_size > 1))

Under PP2 that turns sequence parallelism off, which is correct, and which this
configuration does not use anyway -- it sets no expert parallelism. A check that
cannot tell a gate from a guard is worse than no check: it fails loudly on a
healthy image, and it did, at 90 seconds a job across six jobs.

So walk the syntax instead and look only for the two forms that actually stop a
run: an ``assert`` on the comparison, or an ``if`` whose body raises.
"""

import ast
import importlib.util
import pathlib

ROOT = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
K3 = ROOT / "models/kimi_k3/nvidia"


def refusals(names: list[str]) -> list[tuple[str, int, str]]:
    """Every assert-or-raise in the K3 model files guarding one of `names`."""
    out = []
    for path in sorted(K3.glob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                test = node.test
            elif isinstance(node, ast.If) and any(
                isinstance(b, ast.Raise) for b in node.body
            ):
                test = node.test
            else:
                continue
            rendered = ast.unparse(test)
            if any(n in rendered for n in names):
                out.append((path.name, node.lineno, rendered))
    return out


def main() -> int:
    pp = refusals(["pipeline_parallel_size"])
    assert not pp, (
        "this image refuses pipeline parallelism for Kimi-K3 and the run would "
        f"die at engine init: {pp}"
    )

    # The PCP refusal IS expected to be here. If it ever disappears that is worth
    # knowing -- it is the one parallelism axis this model has never been able to
    # use -- but it must not be read as licence to set PCP.
    pcp = refusals(["prefill_context_parallel_size"])

    # And both halves of dspark-pp-828's own guard. The pp>2 refusal is
    # load-bearing on the speculative arms: a middle stage has never run on
    # hardware and fails as degraded acceptance rather than a crash, which under
    # synthetic rejection -- where the run reports the acceptance it was handed --
    # would be invisible. pp=2 has no middle stage.
    mr = (ROOT / "v1/worker/gpu/model_runner.py").read_text()
    assert "supports_aux_hidden_states_over_pp" in mr, (
        "the narrowed PP+spec refusal is missing; the blanket one may have been "
        "removed without its replacement"
    )
    assert "pipeline_parallel_size=2" in mr, (
        "the pp>2 refusal is gone; a middle stage could run and, under synthetic "
        "acceptance, fail without showing it"
    )

    print("=== pp2 preflight: no PP refusal; pp=2 spec accepted, pp>2 refused; "
          f"PCP still refused ({len(pcp)} sites) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
