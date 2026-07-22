# SPDX-License-Identifier: Apache-2.0
"""
Route eagle/spec decode queries (q_len>1) to the DCP-aware PREFILL attention
path under decode-context-parallelism, by setting supports_dcp_with_varlen=False
on the MLA metadata builders.

Why: under DCP the *decode* attention kernel for a spec query (q_len = num_spec+1)
over the DCP-interleaved local KV shard uses an end-aligned causal mask that is
WRONG for q_len>1 (each spec token misses (dcp_world_size-1)*(q_len-1-i) KV
entries), because the decode kernel receives no cp_rank/global-seq-len info. The
non-MLA FlashInfer backend already sets supports_dcp_with_varlen=False for exactly
this reason (see vllm/v1/attention/backends/flashinfer.py) so that the reorder
threshold stays at 1 and spec queries take the CP-aware prefill path. But
tokenspeed_mla sets it True (and our PR48180-on-94c0ef30 patch set flashinfer_mla
True too) -> spec queries take the decode path -> the target's spec verification
mask is wrong -> the draft is rejected ~always -> real acceptance length ~1.0.

This patch flips both MLA builders to False so spec routes to the DCP-aware
prefill path. Gated on APPLY_DCP_SPEC_PREFILL. Verifies the hypothesis that the
dcp4 eagle AL~1.0 is the wrong-decode-mask-under-DCP issue.
"""
import importlib.util
import re
import sys
from pathlib import Path

TARGETS = ["v1/attention/backends/mla/tokenspeed_mla.py",
           "v1/attention/backends/mla/flashinfer_mla.py"]


def _vllm_root():
    spec = importlib.util.find_spec("vllm")
    if spec and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            if (Path(root) / "v1").exists():
                return Path(root)
    for py in ("python3.12", "python3.11", "python3.10"):
        c = Path(f"/usr/local/lib/{py}/dist-packages/vllm")
        if c.exists():
            return c
    return None


def main():
    root = _vllm_root()
    if root is None:
        print("[dcp-spec-prefill] vllm not found; skip", file=sys.stderr)
        return
    for rel in TARGETS:
        f = root / rel
        if not f.exists():
            print(f"[dcp-spec-prefill] {rel} absent; skip", file=sys.stderr)
            continue
        t = f.read_text()
        n = t.count("supports_dcp_with_varlen=True")
        if n == 0:
            print(f"[dcp-spec-prefill] {rel}: no True to flip (already False?)", file=sys.stderr)
            continue
        t = t.replace("supports_dcp_with_varlen=True", "supports_dcp_with_varlen=False")
        f.write_text(t)
        print(f"[dcp-spec-prefill] {rel}: flipped {n} supports_dcp_with_varlen True->False", file=sys.stderr)


if __name__ == "__main__":
    main()
