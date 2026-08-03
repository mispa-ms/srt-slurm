#!/bin/bash
# Kimi-K3: enable DSpark speculative decoding together with pipeline parallelism.
# =============================================================================
# Applies vllm-project/vllm#50514 ("Feat/spec decode under pipeline parallel")
# on top of the container's vLLM. Upstream branch: misunp/k3-dspark-pp-v2,
# based on vllm-project/vllm 38a466e7b6.
#
# WHY: vLLM rejects eagle3 / dflash / dspark whenever pipeline_parallel_size > 1.
#   The drafter runs on the last PP stage but consumes auxiliary hidden states
#   tapped from target layers on earlier stages, and those were dropped at the
#   stage boundary. #50514 forwards them and lifts the guard, and fixes the
#   latent defects the guard had been hiding — most importantly a sampled-token
#   broadcast width mismatch that hung every PP + spec run on its first prefill.
#
# Kimi-K3 is the only model that opts in (supports_aux_hidden_states_over_pp);
#   EAGLE3 and dflash under PP stay gated. The PR caps pipeline_parallel_size
#   at 2, which covers our TP4xPP2 validation topology.
#
# SCOPE: AGGREGATED serving only. NIXL PD-disaggregation still hard-blocks PP>1
#   with hybrid KV layouts (HMA) in kv_connector/v1/nixl/base_worker.py, and K3
#   always requires HMA because its KDA layers use MambaSpec.
#
# VERIFICATION: the relay does not fail loudly — a drafter fed missing or
#   mis-ordered taps still emits valid-looking proposals that just get rejected
#   more often. Under our standard `synthetic` rejection sampling the acceptance
#   length is pinned to the configured value, so it hides that entirely. Judge a
#   run with `rejection_sample_method: standard` and compare the per-position
#   acceptance profile against the PP=1 baseline.
#
# The patch refuses to apply to an unexpected tree rather than half-applying.
# =============================================================================

set -euo pipefail

dist_packages=/usr/local/lib/python3.12/dist-packages
patch_file=/configs/patches/k3-dspark-pp-aux-relay.patch

guard_target="${dist_packages}/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
relay_target="${dist_packages}/vllm/models/kimi_k3/nvidia/model.py"

unpatched_marker="DSpark does not support pipeline parallelism."
patched_marker="supports_aux_hidden_states_over_pp"

for f in "${guard_target}" "${relay_target}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: ${f} not found; container layout changed" >&2
        exit 1
    fi
done

if grep -Fq "${unpatched_marker}" "${guard_target}"; then
    patch --batch --forward -p1 -d "${dist_packages}" < "${patch_file}"
elif grep -Fq "${patched_marker}" "${relay_target}"; then
    echo "K3 DSpark PP aux-relay patch is already applied"
else
    echo "ERROR: unexpected dspark/utils.py; refusing to patch" >&2
    exit 1
fi

# Verify both halves landed: the guard must be gone AND the relay present.
# Checking only one would let a partially-applied patch through.
if grep -Fq "${unpatched_marker}" "${guard_target}"; then
    echo "ERROR: DSpark PP guard still present after patch" >&2
    exit 1
fi
if ! grep -Fq "${patched_marker}" "${relay_target}"; then
    echo "ERROR: K3 aux relay missing after patch" >&2
    exit 1
fi

# A successful text edit that leaves an unimportable module would otherwise
# surface as a server crash minutes later.
python3 - <<'PYCHK' || { echo "ERROR: patched vLLM does not import cleanly" >&2; exit 1; }
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.v1.worker.gpu import pp_utils  # noqa: F401
assert hasattr(EagleModelMixin, "supports_aux_hidden_states_over_pp")
print("import check OK")
PYCHK

echo "K3 DSpark PP aux-relay patch applied"
