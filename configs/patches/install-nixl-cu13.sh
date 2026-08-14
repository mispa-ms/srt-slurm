#!/bin/bash
# shellcheck shell=bash
#
# NIXL for the stock vLLM nightly, shared by both GB300 setup scripts.
#
# The stock nightly's NIXL cannot register VRAM on oci-aga:
#   ucx_utils.cpp:622  4 NVIDIA GPU(s) were detected, but UCX CUDA support was
#                      not found! GPU memory is not supported.
#   nixl_agent.cpp:468 registerMem: registration failed ... all potential backends
#   nixl_cu13._bindings.nixlBackendError: NIXL_ERR_BACKEND
# Our source-built images never hit this, because AIB builds them with
# INSTALL_KV_CONNECTORS=true and vLLM's Dockerfile then installs
# requirements/kv_connectors.txt (which pins nixl==1.3.1) before force-installing
# nixl-cu<major> over it.
#
# Version, and why not the pinned one: the cu13 wheel's bundled UCX has shipped
# as nixl-cu12's build. auditwheel names libraries by content hash, so it is
# checkable:
#
#   nixl-cu13==1.3.0  libucp-fb7bfdea.so.0.0.0
#   nixl-cu13==1.3.1  libucp-fb7bfdea.so.0.0.0   <- same as cu12
#   nixl-cu13==1.3.2  libucp-e76cb9e6.so.0.0.0
#   nixl-cu12==1.3.1  libucp-fb7bfdea.so.0.0.0
#
# Hanjie's working container carries 1.3.2. Following
# .buildkite/scripts/install-kv-connectors.sh to the letter is what pinned us to
# 1.3.1, since it takes the version from the generic `nixl` requirement.
set -euxo pipefail

echo "=== installing NIXL ${NIXL_VER:-1.3.2} ==="
python3 -m pip install --no-cache-dir "nixl==${NIXL_VER:-1.3.2}"
KV_META=$(python3 -c "
import importlib.metadata as md
import torch
cuda = torch.version.cuda
if cuda is None:
    raise SystemExit('torch.version.cuda is not set')
print(cuda.split('.', 1)[0], md.version('nixl'))
")
read -r CU_MAJOR NIXL_VERSION <<<"${KV_META}"

# Leave the other cuda variant alone. install-kv-connectors.sh removes it ("Keep
# only the variant matching this CI image"), and Hanjie's working container has
# both nixl_cu12.libs and nixl_cu13.libs -- which is simply what vLLM's
# Dockerfile produces, since it force-reinstalls over `nixl` without uninstalling
# anything.
#
# It makes no difference to which one loads, and the earlier note here claiming
# it did was wrong. nixl/__init__.py dispatches on torch's CUDA major:
#
#     def _load_cuda_backend():
#         cuda_major = _get_torch_cuda_major()
#         ...  importlib.import_module(f"nixl_cu{cuda_major}")
#
# With a cu13 torch, nixl_cu12 is never imported no matter what is installed. So
# removing it was not the cause of the oci-aga failure and keeping it is not the
# cure -- it is kept only because matching the working artefact costs nothing.
python3 -m pip install --no-cache-dir --force-reinstall --no-deps "nixl-cu${CU_MAJOR}==${NIXL_VERSION}"
python3 -c "
import importlib.metadata as md
import pathlib
import nixl
present = sorted('%s==%s' % (d.metadata['Name'], d.version)
                 for d in md.distributions()
                 if (d.metadata['Name'] or '').startswith('nixl'))
print('  nixl distributions:', present)
print('  dispatched to:', nixl._bindings.__file__)
cu = [p for p in present if p.startswith('nixl-cu')]
assert any(p.startswith('nixl-cu13') for p in cu), (
    'nixl-cu13 is missing; UCX will have no CUDA-13 component: %s' % present)
"
