#!/usr/bin/env bash
# Kimi-K3 on Hecate / VR200: the bring-up probe, then narrow the two arch gates
# that let Rubin into CuTe DSL kernels that cannot be compiled for it.
#
# ── WHAT BREAKS, EXACTLY ─────────────────────────────────────────────────────
# Pipeline 64804977 reached model construction on the stock nightly and died in
# every rank with
#
#     File ".../kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/
#            allreduce_rmsnorm_reduce_scatter_early_exit.py", line 809
#       _COMPILED[key] = cute.compile(
#     File ".../cutlass/base_dsl/arch.py", line 106, in from_string
#       return cls[arch_str]
#     KeyError: 'sm_107a'
#
# The architecture is otherwise fine. That same run had already logged
#     [mxfp4.py:655] Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend.
#     [kda.py:314]   Using FlashKDA KDA prefill backend.
#     [kda.py:483]   Fused KDA decode kernel (conv+KDA+norm) is enabled.
# so MXFP4 MoE and KDA both resolve on sm_107. CUTLASS DSL's Arch enum simply
# has no sm_107a, and only the CuTe DSL kernels go through it.
#
# ── WHY RUBIN IS LET IN ──────────────────────────────────────────────────────
# platforms/interface.py defines
#     is_device_capability_family(c) -> (current // 10) == (c // 10)
# so family(100) means "any 10.x" -- and 10.7 // 10 == 10. Every K3 site that
# means "SM100 only" spells it this way, including the two below whose comments
# explicitly say SM100 and promise a fallback.
#
# ── WHAT IS PATCHED, AND WHAT DELIBERATELY IS NOT ────────────────────────────
# Five sites in the K3 plugin use the coarse gate. Only two guard a CuTe DSL
# compile, and only those two are touched:
#
#   PATCHED  latent_moe_runner.py:77  -> latent-MoE tail fusion. The confirmed
#            killer above. No env var can disable it: the
#            VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION our B300 configs carry does
#            not appear anywhere at this sha (git grep, zero matches), and TP=8
#            is inside the runner's supported (8, 16).
#
#   PATCHED  model.py:174 -> maybe_init_gemm_rs_ar, which reaches
#            cute_dsl/gemm_rs_ar.py:680 cute.compile. VLLM_KIMI_K3_GEMM_AR
#            defaults to TRUE, so this fires unasked, and its safety net is
#            `except RuntimeError` -- a KeyError is not a RuntimeError and would
#            not be caught. We have not hit it only because the latent-MoE tail
#            dies first. Patching rather than setting VLLM_KIMI_K3_GEMM_AR=0
#            keeps this config's environment identical to the B300 baseline's,
#            so the only difference between the two arms stays "Rubin takes the
#            documented fallback" rather than "we turned a feature off".
#
#   LEFT     kda.py:170 -> guards torch.ops._C.fused_kda_decode, a compiled C++
#            op behind a hasattr() belt, and CUDA 13 family binaries genuinely
#            do run across 10.x. Its own comment says so, and the run log proves
#            it: the fused KDA decode kernel came up enabled on sm_107.
#
#   LEFT     ops/attn_res.py:199 -> same shape, torch.ops._C.kimi_k3_attn_res
#            behind hasattr(). A C++ family binary, not CuTe DSL.
#
#   LEFT     ops/latent_moe_tail.py:120 -> the op's own `[0] != 10` guard. Once
#            the runner gate is narrowed this is unreachable, and it is a
#            defensive check, so leave it where the author put it.
#
# is_device_capability_family itself is NOT touched. Core vLLM uses it in ~20
# files for FlashInfer/TRTLLM paths that demonstrably work on sm_107 -- the
# MXFP4 MoE backend above is one of them. Narrowing it globally would break
# what currently works.
#
# ── IF A LATER NIGHTLY KNOWS sm_107a ─────────────────────────────────────────
# Drop this script and MEASURE the fused path. Do not assume it is faster on
# Rubin just because it is on Blackwell.

set -euo pipefail

bash /configs/patches/kimi-k3-hecate-probe.sh

echo "=== narrowing the K3 CuTe DSL arch gates for sm_107 ==="

python3 - <<'PY'
import pathlib
import sys

import vllm

ROOT = pathlib.Path(vllm.__file__).parent
MARK = "PATCHED-kimi-k3-hecate-rubin"

# (path, old, new) -- both replacements are the same one-line idea: an exact
# capability where the source said "family", which is what the surrounding
# comments already claim the kernels require.
EDITS = [
    (
        "models/kimi_k3/nvidia/latent_moe_runner.py",
        """        self.enable_k3_latent_moe_tail_fusion = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        )""",
        """        # """ + MARK + """: exact capability, not the 10.x family.
        # These kernels are tcgen05 CuTe DSL and CUTLASS has no sm_107a.
        _cap = current_platform.get_device_capability()
        self.enable_k3_latent_moe_tail_fusion = (
            current_platform.is_cuda()
            and _cap is not None
            and _cap.to_int() in (100, 103)
        )""",
    ),
    (
        "models/kimi_k3/nvidia/model.py",
        """    elif not current_platform.is_device_capability_family(100):
        reason = "the device is not SM100-family\"""",
        """    elif (  # """ + MARK + """: exact capability, not the 10.x family.
        current_platform.get_device_capability() is None
        or current_platform.get_device_capability().to_int() not in (100, 103)
    ):
        reason = "the device is not SM100 or SM103\"""",
    ),
]

changed = 0
for rel, old, new in EDITS:
    target = ROOT / rel
    if not target.is_file():
        sys.exit(f"FATAL: {target} not found; the K3 plugin layout changed.")
    src = target.read_text()
    if MARK in src:
        print(f"already patched: {rel}")
        continue
    if old not in src:
        sys.exit(
            f"FATAL: {rel} does not contain the text this patch was written "
            "against. The plugin moved; re-derive the patch rather than "
            "letting it silently no-op."
        )
    target.write_text(src.replace(old, new, 1))
    print(f"patched {rel}")
    changed += 1

print(f"{changed} file(s) changed")

# Say out loud what the gates now decide on THIS device, before eight ranks
# spend forty minutes finding out.
from vllm.platforms import current_platform

cap = current_platform.get_device_capability()
cap_int = cap.to_int() if cap is not None else None
eligible = current_platform.is_cuda() and cap_int in (100, 103)
print(f"device capability = {cap_int}")
print(f"  latent-MoE tail fusion : {'ENABLED' if eligible else 'disabled (fallback)'}")
print(f"  GEMM-AR / GEMM-RS      : {'ELIGIBLE' if eligible else 'disabled (fallback)'}")
if eligible:
    print("NOTE: this device keeps both fused paths -- correct on B200/B300, "
          "and it means the patch changed nothing here.")
PY

echo "=== gates narrowed ==="

# ── A ptxas that knows sm_107 ────────────────────────────────────────────────
# After the gates above, pipeline 64809381 loaded the model (192.85 GiB/rank,
# 176 s) and died in memory profiling on
#
#     ptxas fatal : Value 'sm_107a' is not defined for option 'gpu-name'
#     Repro: /usr/local/cuda/bin/ptxas --gpu-name=sm_107a /tmp/x.ptx -o ...
#
# This is the CUDA toolchain, not vLLM. The container ships CUDA 13.0.88, whose
# ptxas has no sm_107 at all. Checked, by pulling the aarch64 wheels and reading
# the target table out of each binary:
#
#     13.0.88   sm_100 100a 100f 103 103a 103f 120 121 ...      NO sm_107
#     13.3.73   ... adds sm_110 110a 110f ...                   NO sm_107
#     13.4.46rc1  sm_100 ... 103 ... sm_107 sm_107a sm_107f ... YES
#
# 13.4 is exactly the base the framework team's Rubin images are built on --
# ci/sweep_configs/hecate-testing/sglang/gpt-oss-120b/VR200/AGG/... says
# "CUDA 13.4 inference-devel base", and that image is the one that produced the
# only successful sweep_hecate we can find (pipeline 57014089, gpt-oss-120b,
# 2026-07-06). So install 13.4's ptxas and point Triton at it. TRITON_PTXAS_PATH
# is the documented knob for exactly this -- vLLM's own troubleshooting guide
# recommends it when "the ptxas in the triton bundle is not compatible with your
# device".
#
# The variable itself is set in the config's aggregated_environment, NOT here:
# this script runs as a child bash, so an export would not reach the server.
# This step only has to make the binary exist at the path the config names.
#
# CAVEAT, recorded because it may be the next wall. The same failure logged
#     'sm_107a' is not a recognized processor for this target (ignoring processor)
# from LLVM, i.e. Triton's own backend does not know sm_107a either and fell
# back to a default target before handing the PTX to ptxas. A newer ptxas fixes
# the hard error; whether the PTX LLVM produced is right for Rubin is a separate
# question. For the plain elementwise kernels this stack needs, it should be --
# everything arch-specific here goes through FlashInfer/CUTLASS cubins, which
# already work on sm_107. If Triton kernels compile but misbehave, that is where
# to look.
PTXAS_PKG_VER="${K3_PTXAS_VER:-13.4.46rc1}"
echo "=== installing ptxas from nvidia-cuda-nvcc==${PTXAS_PKG_VER} ==="
python3 -m pip install --no-deps --quiet "nvidia-cuda-nvcc==${PTXAS_PKG_VER}"

python3 - <<'PY'
import pathlib
import subprocess
import sys
import sysconfig

site = pathlib.Path(sysconfig.get_paths()["purelib"])
ptxas = site / "nvidia/cu13/bin/ptxas"
if not ptxas.is_file():
    hits = list(site.glob("nvidia/**/bin/ptxas"))
    sys.exit(f"FATAL: ptxas not at {ptxas}; found instead: {hits}")

ptxas.chmod(0o755)
ver = subprocess.run([str(ptxas), "--version"], capture_output=True, text=True)
print(ver.stdout.strip() or ver.stderr.strip())

# The whole point is sm_107. Prove this binary accepts it rather than assuming
# the version number implies it.
probe = pathlib.Path("/tmp/_sm107_probe.ptx")
probe.write_text(
    ".version 8.0\n.target sm_90\n.address_size 64\n"
    ".visible .entry _nop() { ret; }\n"
)
for arch in ("sm_107a", "sm_107"):
    r = subprocess.run(
        [str(ptxas), f"--gpu-name={arch}", str(probe), "-o", "/tmp/_probe.o"],
        capture_output=True, text=True,
    )
    ok = "accepts" if "is not defined for option" not in r.stderr else "REJECTS"
    print(f"  {ptxas.name} {ok} --gpu-name={arch}")
    if ok == "REJECTS":
        sys.exit(f"FATAL: this ptxas does not know {arch}: {r.stderr.strip()[:200]}")

print(f"ptxas ready at {ptxas}")
print("  (the config must set TRITON_PTXAS_PATH to exactly this path)")
PY

echo "=== ptxas ready ==="
