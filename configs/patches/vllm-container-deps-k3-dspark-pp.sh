#!/bin/bash
# Kimi-K3: enable DSpark speculative decoding together with pipeline parallelism.
# =============================================================================
# WHY: vLLM rejects `method: dspark` whenever pipeline_parallel_size > 1. The
#   drafter consumes EAGLE-3 auxiliary hidden states collected at several target
#   layers (Kimi-K3-DSpark: target_layer_ids [2,23,47,71,89] -> installed as aux
#   layers (3,24,48,72,90) after vLLM's +1 conversion). Under PP those layers sit
#   on different ranks, and a non-last rank returned only hidden_states/residual,
#   so the aux collected upstream was dropped and the drafter — built only on the
#   last stage — saw a partial feature set. Two guards enforced the rejection:
#       vllm/v1/worker/gpu/model_runner.py            ValueError for eagle3/dflash/dspark + PP
#       vllm/v1/worker/gpu/spec_decode/dspark/utils.py NotImplementedError
#
# FIX: relay each rank's aux tensors to the next stage inside IntermediateTensors
#   under aux_0..aux_n keys, and gate the runner's PP rejection on a per-model
#   capability flag that only Kimi-K3 declares. Other models stay rejected.
#   Six files, pure Python — no rebuild, no compiled extension touched.
#
#   Upstream branch: misunp/k3-dspark-pp, based on vllm-project/vllm 38a466e7b6.
#   Design: docs/superpowers/specs/2026-08-01-k3-dspark-pp-design.md
#
# SCOPE: this enables PP>1 for AGGREGATED serving only. NIXL PD-disaggregation
#   still hard-blocks PP>1 with hybrid KV layouts (HMA) in
#   kv_connector/v1/nixl/base_worker.py — untouched here, and K3 always requires
#   HMA because its KDA layers use MambaSpec.
#
# VERIFICATION NOTE: correctness of the relay is only visible through acceptance
#   length under `rejection_sample_method: standard`. Under our standard
#   `synthetic` setting, AL is pinned to the configured value and output accuracy
#   is deliberately degraded, so neither signals a broken aux path. Run the
#   validation config with `standard` before trusting a `synthetic` perf number.
#
# The patch refuses to apply to an unexpected tree rather than half-applying: a
#   silently partial aux relay degrades acceptance length without crashing, which
#   is exactly the failure mode that hides in a benchmark.
# =============================================================================

# pp_aux_split is delivered by APPEND, not by the patch. It is a standalone
# function that lands at the end of models/utils.py, so a context diff anchors on
# whatever happens to be the last function in the container's copy — which drifts.
# Pipeline 60696412 failed exactly there ("1 out of 1 hunk FAILED") while the other
# five files applied at offset -19/-20. Appending is immune to that drift.

set -euo pipefail

dist_packages=/usr/local/lib/python3.12/dist-packages
patch_file=/configs/patches/k3-dspark-pp-aux-relay.patch
aux_split_file=/configs/patches/k3_pp_aux_split.py

guard_target="${dist_packages}/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
relay_target="${dist_packages}/vllm/models/kimi_k3/nvidia/model.py"
utils_target="${dist_packages}/vllm/model_executor/models/utils.py"

unpatched_marker="DSpark does not support pipeline parallelism."
patched_marker="supports_pp_aux_hidden_states"
aux_split_marker="def pp_aux_split("

for f in "${guard_target}" "${relay_target}" "${utils_target}"; do
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

if grep -Fq "${aux_split_marker}" "${utils_target}"; then
    echo "pp_aux_split already present"
else
    printf '\n\n' >> "${utils_target}"
    cat "${aux_split_file}" >> "${utils_target}"
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
if ! grep -Fq "${aux_split_marker}" "${utils_target}"; then
    echo "ERROR: pp_aux_split missing after append" >&2
    exit 1
fi

# The relay imports pp_aux_split; a successful text edit that leaves an
# unimportable module would surface as a server crash minutes later instead.
python3 -c "from vllm.model_executor.models.utils import pp_aux_split" || {
    echo "ERROR: pp_aux_split is not importable after patching" >&2
    exit 1
}

echo "K3 DSpark PP aux-relay patch applied"
