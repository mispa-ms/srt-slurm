#!/bin/bash
# shellcheck shell=bash
#
# Read the run that works instead of reconstructing it.
#
# Hanjie Qiu reproduces Wei's numbers on this cluster, in a container built the
# same way, while every arm of ours dies registering VRAM through NIXL. Eight
# submits have now gone into guessing which difference matters -- module dir,
# wheel version, cuda variant, peermem, dmabuf, UCX_TLS -- one hypothesis per
# twelve-minute cycle, each answered and replaced by the next.
#
# His logs are on this filesystem, and this script runs on this filesystem.
# srt-slurm writes the full per-worker environment and command into every sweep
# log ("Env:" / "Command:" at INFO), so his working configuration is readable
# rather than inferable. Diff ours against his and stop guessing.
#
# Writes to $UCX_DIFF_OUT rather than stdout. srt-slurm quotes only the LAST 50
# lines of a failed process log, so anything printed before the probe's exit is
# guaranteed to be pushed out of the window -- which is what happened to the
# first version of this: it ran, found its answer, and none of it survived.
# The caller prints this file last.
set -uo pipefail

exec >"${UCX_DIFF_OUT:-/dev/stdout}" 2>&1

# Every user, not one. Betting on a name is how this started: the first version
# looked only at hanjieq, and AIB shows no successful oci-aga run by anyone --
# 30 sweep jobs on this cluster, all ours, all failed or cancelled. But
# srt-slurm is also driven by hand here (Zachary Patel: "I sent a test job in
# with srt slurm for Minimax on oci-aga and it worked"), and those runs land in
# the same place under a different user. Glob the users directory instead of
# guessing usernames.
USER_GLOBS="
/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/*/srt-slurm/outputs
/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/*/srt-slurm/outputs
"

echo "=== the run that works ==="
_roots=""
for _g in ${USER_GLOBS}; do
    for _r in ${_g}; do
        [ -d "${_r}" ] && _roots="${_roots} ${_r}"
    done
done
if [ -z "${_roots}" ]; then
    echo "  no srt-slurm outputs under any user on this cluster"
    exit 0
fi
echo "  searching $(echo ${_roots} | wc -w) user output trees"

# A run that worked is one whose workers instantiated the UCX backend. The
# failure we are chasing prints "UCX CUDA support was not found" instead, so
# that string is the discriminator, not the job's exit status -- a run can fail
# for unrelated reasons with a perfectly good UCX.
_good=""
_root=""
for _r in ${_roots}; do
    for _job in $(ls -1t "${_r}" 2>/dev/null | head -15); do
        _logs="${_r}/${_job}/logs"
        [ -d "${_logs}" ] || continue
        if grep -rlq "Backend UCX was instantiated" "${_logs}" 2>/dev/null &&
           ! grep -rq "UCX CUDA support was not found" "${_logs}" 2>/dev/null; then
            _good="${_job}"; _root="${_r}"
            break 2
        fi
    done
done
if [ -z "${_good}" ]; then
    echo "  no run under any user shows a working UCX backend"
    for _r in ${_roots}; do
        echo "    $(echo "${_r}" | sed 's|.*/users/||; s|/srt-slurm.*||'): $(ls -1 "${_r}" 2>/dev/null | wc -l) jobs"
    done
    exit 0
fi
echo "  ${_root}"
echo "  job ${_good}"

_theirs=$(mktemp)
_ours=$(mktemp)

# srt-slurm logs "Env: A=1 B=2 ..." once per worker. Take the first worker's.
grep -rhom1 "Env: .*" "${_root}/${_good}/logs" 2>/dev/null | head -1 \
    | sed 's/^Env: //' | tr ' ' '\n' | grep "=" | sort -u >"${_theirs}"
env | sort -u >"${_ours}"

echo "  their env: $(wc -l <"${_theirs}") vars"

echo "=== variables they set and we do not (or set differently) ==="
# Restricted to the families that can plausibly move NIXL/UCX. A full env diff
# is dominated by SLURM_*, hostnames and paths that differ by construction.
_keys="UCX NIXL NCCL CUDA NVIDIA LD_ VLLM MOONCAKE GLOO TORCH"
_found=0
while IFS= read -r _line; do
    _name=${_line%%=*}
    case "${_name}" in
        *UCX*|*NIXL*|*NCCL*|*CUDA*|*NVIDIA*|LD_*|*VLLM*|*MOONCAKE*|*GLOO*|*TORCH*) ;;
        *) continue ;;
    esac
    _ourval=$(grep -m1 "^${_name}=" "${_ours}" | cut -d= -f2-)
    _theirval=${_line#*=}
    if [ "${_ourval}" != "${_theirval}" ]; then
        echo "  ${_name}"
        echo "      theirs: ${_theirval}"
        echo "      ours:   ${_ourval:-<unset>}"
        _found=$((_found + 1))
    fi
done <"${_theirs}"
[ "${_found}" -eq 0 ] && echo "  (none -- the environments agree on every UCX/NIXL/CUDA variable)"

echo "=== variables we set and they do not ==="
while IFS= read -r _line; do
    _name=${_line%%=*}
    case "${_name}" in
        *UCX*|*NIXL*|*MOONCAKE*) ;;
        *) continue ;;
    esac
    grep -q "^${_name}=" "${_theirs}" || echo "  ${_name}=${_line#*=}"
done <"${_ours}"

echo "=== their container ==="
grep -rhom1 "container_image[^ ]*[:=] *[^ ]*" "${_root}/${_good}"/*.log "${_root}/${_good}"/logs/*.log 2>/dev/null | sort -u | head -3
grep -rhoE "\-\-container-image[= ][^ ]+" "${_root}/${_good}/logs" 2>/dev/null | sort -u | head -3

echo "=== what UCX chose in their run ==="
grep -rhoE "(Backend UCX was instantiated|selected transport[s]?[^,]*|using dmabuf[^,]*)" \
    "${_root}/${_good}/logs" 2>/dev/null | sort -u | head -5
