#!/bin/bash
# shellcheck shell=bash

# --- why UCX may not see CUDA -------------------------------------------
# nixl's wheel does ship the CUDA plugins (libuct_cuda.so, libucm_cuda.so), so
# "UCX CUDA support was not found" is a load failure, not a missing build. Their
# NEEDED list is the tell:
#   libuct_cuda.so -> libcuda.so.1, libnvidia-ml.so.1
#   libucm_cuda.so -> libcuda.so.1, libcudart.so.13
# torch only needs libcuda + libcudart, which is why the model loads and KV cache
# is sized normally while UCX still refuses VRAM. libnvidia-ml.so.1 is injected by
# the container runtime only when NVIDIA_DRIVER_CAPABILITIES includes `utility`.
# Print the loader's view so the next failure names the missing library instead
# of leaving us to infer it.
echo "=== driver libraries the UCX CUDA plugin needs ==="
echo "  NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
for _lib in libcuda.so.1 libnvidia-ml.so.1 libcudart.so.13; do
    _found=$(ldconfig -p 2>/dev/null | grep -m1 "${_lib}" || true)
    if [ -n "${_found}" ]; then
        echo "  ok      ${_lib}: ${_found##*=> }"
    else
        echo "  MISSING ${_lib}"
    fi
done
# Resolve the plugin dir from the library the loader actually binds, not from a
# glob. With both cuda variants installed, `glob('nixl*.libs/ucx')` returns
# nixl_cu12 first purely because it sorts first, which is what made the previous
# guard kill every oci arm: it compared a correct UCX_MODULE_DIR against the
# wrong sibling. ldd on the binding extension resolves the libucs that will be
# loaded, and its own directory is the only one that matters.
_plugdir=$(python3 -c "
import pathlib, subprocess
try:
    import nixl
    binding = nixl._bindings
except Exception:
    raise SystemExit(0)
print('  binding:', binding.__file__)
out = subprocess.run(['ldd', binding.__file__], capture_output=True, text=True).stdout
for line in out.splitlines():
    if 'libucs' in line and '=>' in line:
        print(pathlib.Path(line.split('=>')[1].split('(')[0].strip()).parent / 'ucx')
        break
" 2>/dev/null | tee /dev/stderr | tail -1)
if [ -n "${_plugdir}" ]; then
    echo "  ucx plugin dir (from the loaded libucs): ${_plugdir}"
    for _p in "${_plugdir}"/libuct_cuda.so "${_plugdir}"/libucm_cuda.so; do
        [ -e "${_p}" ] && echo "    present: $(basename ${_p})"
    done
    # ldd resolves exactly what the loader will, including the injected driver.
    ldd "${_plugdir}/libuct_cuda.so" 2>/dev/null | grep -E "not found|libcuda|libnvidia-ml" | sed 's/^/    /' || true
    # UCX_MODULE_DIR should NOT be set. libucs carries the compiled-in default
    # "/usr/lib64/ucx", which is what we reasoned from, but it does not use it
    # blindly: it exports ucs_sys_get_lib_path and ucs_module_loader_add_dl_dir,
    # and calls dladdr on its own symbol to locate itself, then searches the
    # module dir relative to where it actually lives. auditwheel's relocation is
    # therefore already handled -- the plugins sit next to libucs in the wheel's
    # .libs and UCX finds them.
    #
    # Setting the variable can only do harm here, because both cuda variants are
    # installed: it forces one absolute directory on whichever libucs is loaded,
    # so if the process binds nixl_cu12's libucs it is handed cu13's plugins.
    if [ -n "${UCX_MODULE_DIR:-}" ] && [ "${UCX_MODULE_DIR}" != "${_plugdir}" ]; then
        echo "[ucx] WARNING: UCX_MODULE_DIR=${UCX_MODULE_DIR} overrides the" \
             "dir belonging to the loaded libucs (${_plugdir}). Unset it in the arm." >&2
    fi
    # The cu13 wheel has shipped a cu12 UCX before (1.3.0 and 1.3.1 both bundle
    # nixl-cu12's libucp). Catch that here rather than 40 minutes later as
    # "UCX CUDA support was not found" with every library resolving.
    _ucp=$(ls "${_plugdir}"/../libucp-*.so.* 2>/dev/null | head -1)
    if [ -n "${_ucp}" ]; then
        echo "    bundled UCX: $(basename "${_ucp}")"
        case "$(basename "${_ucp}")" in
            libucp-fb7bfdea.*)
                echo "[ucx] FATAL: this is nixl-cu12's UCX inside the cu13 wheel" \
                     "(nixl-cu13 1.3.0/1.3.1). Its CUDA component will not load on a" \
                     "CUDA 13 image. Pin NIXL_VER=1.3.2 or later." >&2
                exit 1 ;;
        esac
    fi
    # Is there a system UCX in this image at all? The wheel's copy is the only
    # one vLLM's Dockerfile installs, but a base image or a transitive dep (e.g.
    # OpenMPI) can drag one in -- and a properly installed UCX has its CUDA
    # component where its own prefix expects it. If one exists, pointing at it
    # is a better answer than routing NIXL around UCX, which changes the
    # transport and therefore the numbers.
    echo "  system UCX (outside the wheel):"
    ldconfig -p 2>/dev/null | grep -E "libucp|libuct|libucs" | grep -v "nixl" | sed 's/^/    /' || echo "    none"
    for _d in /usr/lib64/ucx /usr/lib/aarch64-linux-gnu/ucx /usr/local/ucx/lib/ucx /opt/hpcx/ucx/lib/ucx; do
        [ -d "${_d}" ] && echo "    module dir exists: ${_d} ($(ls "${_d}" 2>/dev/null | grep -c cuda) cuda modules)"
    done
fi

# --- functional probe -----------------------------------------------------
# Everything above is static: which files exist, what ldd resolves. None of it
# distinguishes "the plugin is present" from "the plugin is present and dlopen
# rejects it", and that distinction is the whole open question on oci-aga. So
# ask UCX to do the thing that fails. UCX_LOG_LEVEL=debug makes the module
# loader print the dlerror for every module it declines, which no amount of
# file listing can give us.
#
# This runs in the same container, on the same node, ~2 minutes in, instead of
# surfacing as NIXL_ERR_BACKEND at KV-cache init ten minutes later.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "=== probing UCX for CUDA support (debug log) ==="
    _ucxlog=$(mktemp)
    UCX_LOG_LEVEL=debug python3 - >"${_ucxlog}" 2>&1 <<'PROBE'
import pathlib

import torch

# Create the CUDA context BEFORE the agent. UCX scans devices when the context
# is built and skips the cuda memory domains if it finds none active -- the
# first version of this probe made the agent first and produced exactly that:
#
#   cuda_ctx.c:30 cuda primary context is inactive on device 0..3
#   ucp_context.c:2073 no memory domain supports registering cuda memory
#
# which says more about the probe's ordering than about the container. vLLM has
# a live context long before NixlConnector builds its agent, so the faithful
# order is this one.
buf = torch.zeros(1024, dtype=torch.uint8, device="cuda:0")
torch.cuda.synchronize()
print(f"PROBE_CTX: cuda context active, {torch.cuda.device_count()} devices")

# What UCX checks for GPUDirect RDMA, reported whichever way the probe goes.
for p in ("/sys/kernel/mm/memory_peers/nv_mem/version",
          "/sys/module/nvidia_peermem/version",
          "/sys/module/nv_peer_mem/version"):
    print(f"PROBE_GDR: {p}: {'present' if pathlib.Path(p).exists() else 'absent'}")

# Load the UCT cuda module by hand. UCX logs its own module loads at debug, but
# the attempt happens during MD resource query, hundreds of lines before the
# failure, and srt-slurm keeps only the tail -- so the one line that matters is
# always off-screen. ctypes gives the dlerror directly, and distinguishes "the
# module will not load" from "it loaded and produced no memory domain", which is
# the open question: ucm's cuda module loads fine, yet UCX reports no MD able to
# register cuda memory.
import ctypes  # noqa: E402
import glob  # noqa: E402

import nixl  # noqa: E402

_libs = pathlib.Path(nixl._bindings.__file__).parent.parent
for _name in ("libuct_cuda.so*", "libuct_ib.so*"):
    _hits = sorted(glob.glob(str(_libs / "*.libs" / "ucx" / _name)))
    if not _hits:
        print(f"PROBE_DLOPEN: {_name}: NOT FOUND under {_libs}")
        continue
    try:
        ctypes.CDLL(_hits[-1], mode=ctypes.RTLD_GLOBAL)
        print(f"PROBE_DLOPEN: {pathlib.Path(_hits[-1]).name}: ok")
    except OSError as exc:
        print(f"PROBE_DLOPEN: {pathlib.Path(_hits[-1]).name}: FAILED: {exc}")

from nixl._api import nixl_agent, nixl_agent_config  # noqa: E402

agent = nixl_agent("ucx-probe", nixl_agent_config(backends=["UCX"]))

# DRAM first: it separates "UCX is broken here" from "UCX works and has no CUDA
# memory type", which are different bugs with different fixes.
host = torch.zeros(1024, dtype=torch.uint8)
try:
    agent.register_memory([host])
    print("PROBE_DRAM_OK: UCX registered host memory")
except Exception as exc:
    print(f"PROBE_DRAM_FAIL: {type(exc).__name__}: {exc}")

agent.register_memory([buf])
print("PROBE_OK: UCX registered VRAM")
PROBE
    _rc=$?
    if [ "${_rc}" -ne 0 ] || ! grep -q PROBE_OK "${_ucxlog}"; then
        # Print the decisive lines LAST and nothing after them. srt-slurm quotes
        # only the "Last 50 lines" of a failed process log, and the first attempt
        # at this dumped `tail -60` of a debug log whose final lines are agent
        # teardown -- so the window showed ucp_ep destroy noise and cut off the
        # module loader entirely. Everything that survives that window has to be
        # signal.
        {
            echo "[ucx] FATAL: UCX cannot register VRAM in this container."
            echo "[ucx] --- python ---"
            grep -vE "UCX +(DEBUG|TRACE|INFO)" "${_ucxlog}" | tail -12
            echo "[ucx] --- probe verdicts ---"
            grep -E "PROBE_" "${_ucxlog}" || echo "  (none reached)"
            # The uct cuda module load happens during MD resource query, hundreds
            # of lines before the failure, so grep for it by name rather than
            # taking a tail that will never reach back that far. The GDR lines
            # repeat once per rail; one is enough.
            # Filter to the cuda components specifically. Taking the first N
            # module.c lines fills the budget with uct_ib/mlx5/efa loads, which
            # are known to work, and never reaches the one that matters. UCX
            # dlopens libuct_cuda.so for the uct component; ctypes shows that
            # library loads fine by hand, so whether UCX asks for it at all is
            # the open half of the question.
            echo "[ucx] --- uct cuda module ---"
            grep -E "loading modules for uct$|module '(cuda|cuda_copy|cuda_ipc|gdr_copy)'|libuct_cuda|cuda_ipc|gdr_copy" \
                "${_ucxlog}" | head -12 || echo "  (uct never asked for a cuda module)"
            echo "[ucx] --- memory domains ---"
            grep -E "memory domain|md open|query .* resources" "${_ucxlog}" | head -8
            # The IB memory domain will register CUDA memory through either
            # nvidia_peermem or dmabuf -- libuct_ib says so itself: "Couldn't
            # enable GPUDirect RDMA. Please make sure nv_peer_mem [...] is
            # installed correctly, or dmabuf is supported." peermem is absent on
            # oci-aga, so whether dmabuf works is the whole question, and UCX
            # already probes it and logs the verdict.
            # What UCX said when the registration itself failed. Every summary so
            # far has described the container's capabilities and none has quoted
            # the error, which is why "dmabuf is supported on every rail" and
            # "registerMem raises NIXL_ERR_BACKEND" can both be true and unmet.
            echo "[ucx] --- registration ---"
            grep -iE "ucx_utils|registerMem|failed to register|reg_mem|memory type|not supported" \
                "${_ucxlog}" | grep -v PROBE_ | tail -12 || echo "  (nothing)"
            echo "[ucx] --- fabric on this node ---"
            grep -oE "(rdma_vf_rail[0-9]+|rdma_rail[0-9]+|mlx5_[0-9]+):" "${_ucxlog}" \
                | sed 's/:$//' | sort -u | tr '\n' ' '; echo
            echo "[ucx] --- dmabuf ---"
            grep -iE "dmabuf" "${_ucxlog}" | head -10 || echo "  (no dmabuf lines)"
            echo "[ucx] --- GPUDirect (one rail) ---"
            grep -E "GPUDirect RDMA is not detected|Couldn't enable GPUDirect" "${_ucxlog}" | head -3
        } >&2
        exit 1
    fi
    echo "  UCX registered VRAM"
    rm -f "${_ucxlog}"
else
    echo "=== skipping UCX probe: no GPU visible in the setup context ==="
fi

