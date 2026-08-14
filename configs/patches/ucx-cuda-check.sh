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
    _probe=$(mktemp /tmp/ucxprobe.XXXXXX.py)
    cat >"${_probe}" <<'PROBE'
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

    # Vary the environment inside one job instead of one hypothesis per submit.
    # The probe is a single process taking about a second, while a submit costs
    # twelve minutes of queue and setup -- and every UCX theory so far has been
    # answered by one run and replaced by the next. Ask all the questions at once.
    #
    # The arm sets UCX_MEMTYPE_CACHE=n and UCX_MEMTYPE_REG_WHOLE=n, both of which
    # touch memory-type handling, and the failure is precisely that UCX reports
    # "VRAM memory is detected as host". That makes them suspects rather than
    # settings, so run with them and without them, and with no UCX_* at all.
    _ucxvars=$(env | sed -n 's/^\(UCX_[A-Z0-9_]*\)=.*/\1/p' | tr '\n' ' ')
    echo "=== probing UCX for CUDA support ==="
    # Values, not just names. UCX_TLS and UCX_NET_DEVICES turned up in this
    # environment even though the arm sets neither and its comment says both are
    # deliberately unset, and neither repo assigns them -- so they arrive from the
    # image or the cluster. A restrictive UCX_TLS is exactly what would leave UCP
    # with no registration-capable memory domain, which is what it reports: not
    # only for cuda, but for host memory too.
    env | grep "^UCX_" | sort | sed 's/^/  env: /' || echo "  env: <no UCX_ vars>"
    _unset_all=""
    for _v in ${_ucxvars}; do _unset_all="${_unset_all} -u ${_v}"; done

    # Two things that would explain the failure without any environment variable
    # being involved, which is what the 'noucx' variant failing implies.
    #
    # A ucx.conf overrides nothing we can unset -- libucs reads /etc/ucx/ucx.conf
    # and one relative to its own install prefix -- so a TLS restriction there
    # survives `env -u`.
    echo "  ucx.conf:"
    for _c in /etc/ucx/ucx.conf "${HOME}/.ucx/ucx.conf" \
              /usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/etc/ucx/ucx.conf; do
        [ -f "${_c}" ] && { echo "    ${_c}:"; sed 's/^/      /' "${_c}"; }
    done
    [ -f /etc/ucx/ucx.conf ] || echo "    (none found)"
    # And without the IB device nodes there is no IB memory domain to have, only
    # tcp -- which registers nothing. That would produce exactly what UCP reports
    # here: no memory domain for host memory either, not just for cuda. The nodes
    # these probes landed on did not all show the same fabric.
    echo "  /dev/infiniband: $(ls /dev/infiniband 2>/dev/null | tr '\n' ' ' || echo '<absent>')"

    _winner=""
    _asislog=""
    _logs=""
    for _variant in asis nomemtype noucx tlsall; do
        _log=$(mktemp)
        case "${_variant}" in
            asis)      env UCX_LOG_LEVEL=debug python3 "${_probe}" >"${_log}" 2>&1 ;;
            nomemtype) env -u UCX_MEMTYPE_CACHE -u UCX_MEMTYPE_REG_WHOLE \
                           UCX_LOG_LEVEL=debug python3 "${_probe}" >"${_log}" 2>&1 ;;
            noucx)     env ${_unset_all} UCX_LOG_LEVEL=debug \
                           python3 "${_probe}" >"${_log}" 2>&1 ;;
            # Override rather than unset. We inherit UCX_TLS=tcp and
            # UCX_NET_DEVICES=eth0 from outside both repos, and tcp registers no
            # memory at all -- but unsetting them did not help, so if a ucx.conf
            # carries the same restriction only an explicit value beats it.
            tlsall)    env UCX_TLS=all UCX_NET_DEVICES=all UCX_LOG_LEVEL=debug \
                           python3 "${_probe}" >"${_log}" 2>&1 ;;
        esac
        # Report every variant's evidence, not just its verdict. The previous
        # round ran four environments and quoted only the first one's log, so
        # 'tlsall: FAILED' carried no information about what changed -- the same
        # mistake as running one hypothesis per submit, just compressed into a
        # single job. Two numbers separate "the override did nothing" from "it
        # opened the fabric and something else stopped it".
        _ibmds=$(grep -c "md open by .* is successful" "${_log}" || true)
        _reason=$(grep -oE "no memory domain supports registering (host|cuda) memory|VRAM memory is detected as host|no selected transport resources" \
                  "${_log}" | sort -u | tr '\n' ';')
        # And the exception, per variant. Unsetting UCX_TLS made the memory-domain
        # messages disappear and the registration still failed, which means the
        # variants fail for different reasons -- so a shared detail section taken
        # from one of them describes the wrong failure for the others.
        # `nixl_cu13._bindings.nixlBackendError` -- the module path carries
        # digits, which [A-Za-z_.] excluded, so every variant reported
        # "<no exception>" while raising one.
        _exc=$(grep -E "^[^ ]+(Error|Exception): " "${_log}" | tail -1)
        if grep -q PROBE_OK "${_log}"; then
            echo "  ${_variant}: VRAM registered (ib mds=${_ibmds})"
            [ -z "${_winner}" ] && _winner="${_variant}"
        else
            echo "  ${_variant}: FAILED (ib mds=${_ibmds}) ${_exc:-<no exception>}"
            echo "      ${_reason:-<no known reason>}"
        fi
        _logs="${_logs} ${_variant}:${_log}"
        [ "${_variant}" = asis ] && _asislog="${_log}"
    done

    if [ -n "${_winner}" ] && [ "${_winner}" != asis ]; then
        echo "[ucx] FATAL: VRAM registration needs the '${_winner}' environment." >&2
        echo "[ucx]   It works without the arm's UCX settings and fails with them." >&2
        echo "[ucx]   arm's UCX vars: ${_ucxvars}" >&2
        echo "[ucx]   Fix the config rather than probing again." >&2
        exit 1
    fi

    if [ -z "${_winner}" ]; then
        # Print the decisive lines LAST and nothing after them. srt-slurm quotes
        # only the "Last 50 lines" of a failed process log, so anything that has
        # to survive must come at the end -- and must be matched by name, never
        # by position: `head`/`tail` on a mixed grep has now silently dropped the
        # one line that mattered three separate times.
        {
            echo "[ucx] FATAL: UCX cannot register VRAM under any tested environment."
            echo "[ucx] --- fabric on this node ---"
            grep -oE "(rdma_vf_rail[0-9]+|rdma_rail[0-9]+|mlx5_[0-9]+):" "${_asislog}" \
                | sed 's/:$//' | sort -u | tr '\n' ' '; echo
            echo "[ucx] --- probe verdicts ---"
            grep -E "PROBE_" "${_asislog}" || echo "  (none reached)"
            for _entry in ${_logs}; do
                _vname=${_entry%%:*}
                _vlog=${_entry#*:}
                echo "[ucx] --- ${_vname} ---"
                _detail=$(grep -hE "no memory domain|detected as host|ucx_utils|^[^ ]+(Error|Exception): |no selected transport" "${_vlog}" \
                          | sed 's/^/      /' | sort -u | head -5)
                if [ -n "${_detail}" ]; then
                    echo "${_detail}"
                else
                    echo "      (no matching line -- the variant failed silently)"
                fi
            done
        } >&2
        exit 1
    fi
    echo "  UCX registered VRAM"
else
    echo "=== skipping UCX probe: no GPU visible in the setup context ==="
fi
