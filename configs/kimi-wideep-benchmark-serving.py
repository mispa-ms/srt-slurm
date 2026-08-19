"""Run the SA random-token benchmark without its optional datasets package."""

import os
import runpy
import sys
import types
from pathlib import Path
from typing import NoReturn


def _unsupported_load_dataset(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("The wide-EP profile workload only supports random prompts")


benchmark_path = Path(
    os.environ.get(
        "KIMI_WIDEEP_SA_BENCHMARK",
        "/srtctl-benchmarks/sa-bench/benchmark_serving.py",
    )
)
datasets_module = types.ModuleType("datasets")
datasets_module.load_dataset = _unsupported_load_dataset
sys.modules["datasets"] = datasets_module
sys.path.insert(0, str(benchmark_path.parent))
if "--skip-initial-test" not in sys.argv:
    sys.argv.append("--skip-initial-test")
runpy.run_path(str(benchmark_path), run_name="__main__")
