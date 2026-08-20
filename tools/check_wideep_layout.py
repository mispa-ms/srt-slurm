"""Simulate the process layout the wide-EP configs will produce, before submitting.

Run from a srt-slurm checkout:  PYTHONPATH=src python tools/check_wideep_layout.py

The schema check passed for a config that then died at
`_dp_per_node_endpoints_to_processes` with dp!=total_gpus. A config can be
schema-valid and still be a topology srtctl cannot lay out, so the gate before a
submit has to build the processes, not just load the YAML.
"""
import sys, yaml, pathlib
sys.path.insert(0, "src")
from srtctl.backends import VLLMProtocol
from srtctl.backends.vllm import VLLMServerConfig
from srtctl.core.topology import Endpoint

cfgdir = pathlib.Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/scratch.misunp_gpu/.aib-wt-wideep/ci/sweep_configs/kimi-k3/vllm/kimi-k3-mxfp4/GB300/AGG"
)
ok = True
for f in sorted(cfgdir.glob("*.yml")):
    d = yaml.safe_load(f.read_text())
    r, b = d["resources"], d["backend"]
    agg = b["vllm_config"]["aggregated"]
    nodes = tuple(f"n{i}" for i in range(r["agg_nodes"]))
    ep = Endpoint(mode="agg", index=0, nodes=nodes,
                  gpu_indices=frozenset(range(r["gpus_per_node"])),
                  gpus_per_node=r["gpus_per_node"])
    backend = VLLMProtocol(dp_launch_mode=b.get("dp_launch_mode", "per_gpu"),
                           vllm_config=VLLMServerConfig(aggregated=dict(agg)))
    # The mode key is "agg"; with "aggregated" get_config_for_mode returns {} and
    # every topology passes vacuously -- which is how a TP8xPP2 arm that srtctl
    # cannot lay out was cleared before submit. Fail loudly rather than silently.
    if not backend.get_config_for_mode("agg"):
        ok = False
        print(f"FAIL {f.stem}\n      backend config did not load for mode 'agg'")
        continue
    tp = agg["tensor-parallel-size"]
    pp = agg.get("pipeline-parallel-size", 1)
    dp = agg.get("data-parallel-size", 1)
    world = tp * pp * dp
    if world != r["gpus_per_agg"]:
        ok = False
        print(f"FAIL {f.stem}\n      TP{tp} x PP{pp} x DP{dp} = {world} != gpus_per_agg {r['gpus_per_agg']}")
        continue
    try:
        procs = backend.endpoints_to_processes([ep])
    except Exception as exc:
        ok = False
        print(f"FAIL {f.stem}\n      {type(exc).__name__}: {exc}")
        continue
    gpus_each = sorted(len(p.gpu_indices) for p in procs)
    print(f"PASS {f.stem}")
    print(f"      TP{tp} x PP{pp} x DP{dp} = {world} over {r['agg_nodes']} nodes / {r['gpus_per_agg']} GPUs"
          f" -> {len(procs)} process(es), GPUs each {gpus_each}, node_ranks {[p.node_rank for p in procs]}")
    if len(procs) != r["agg_nodes"]:
        ok = False
        print(f"      UNEXPECTED: {len(procs)} processes for {r['agg_nodes']} nodes")
print("ALL PASS" if ok else "FAILURES ABOVE")

sys.exit(0 if ok else 1)
