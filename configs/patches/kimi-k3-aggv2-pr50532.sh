#!/usr/bin/env bash
# AGG v2 runtime setup, plus vllm#50532.
#
# #50532 is open, not merged, and Python-only across five files, so it goes on as
# a runtime patch rather than an image rebuild -- the arm using this script and
# the arm using kimi-k3-aggv2.sh run the identical binary otherwise.
#
# WHAT IT FIXES. V2 decides a batch is a uniform decode from its shape alone:
# get_uniform_token_count() returns max_query_len whenever
# num_tokens == max_query_len * num_reqs, with nothing about whether the requests
# are past their prefill. A prompt chunk whose length happens to match therefore
# replays a captured decode graph over prompt tokens. Dense models survive that;
# recurrent-state models do not, because the mis-dispatched batch still builds
# prefill metadata and the graph reads capture-time state indices. K3 is 69/93
# KDA layers.
#
# WHETHER IT REACHES OUR LADDER IS THE OPEN QUESTION. AgentX ISL is 353 tokens
# minimum against a 8192-token budget, so no first chunk is ever clipped to one
# token; what is reachable is a tail chunk of length one sharing a batch with
# decodes. That is why this is priced against a chunking-forced arm and not only
# at stock settings -- see the -chunk16- configs.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-aggv2.sh"

SITE=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
PATCH=/configs/patches/vllm-pr50532-uniform-decode.patch

if patch -p1 -R --dry-run --force --silent -d "$SITE" < "$PATCH" >/dev/null 2>&1; then
    echo "=== vllm#50532 already present, skipping ==="
else
    echo "=== applying vllm#50532 ==="
    patch -p1 --forward -d "$SITE" < "$PATCH"
fi

# Fail the run rather than the analysis: an arm that silently did not get the
# patch would be scored as if it had.
python3 - <<'PY'
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
checks = {
    "v1/worker/utils.py": "def get_uniform_decode_token_count",
    "v1/worker/gpu/cudagraph_utils.py": "get_uniform_decode_token_count",
}
missing = [rel for rel, mark in checks.items()
           if mark not in (root / rel).read_text()]
assert not missing, f"vllm#50532 did not land in: {missing}"
print("=== vllm#50532 verified ===")
PY
