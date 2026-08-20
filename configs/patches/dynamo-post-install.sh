#!/usr/bin/env bash
# Teach dynamo that NixlPushConnector is a PD connector.
# =============================================================================
# WHAT IT FIXES. A disaggregated arm on NixlPushConnector brings both engines
# up, completes the P/D handshake, and then answers every request with HTTP 400.
# The frontend is relaying a prefill failure; the prefill worker's own log says
#
#     ValueError: Unsupported kv_connector='NixlPushConnector' for PD.
#     Supported names: ['MooncakeConnector', 'NixlConnector'].
#
# from components/src/dynamo/vllm/kv_connector_protocols.py. Dynamo keeps its
# own registry of PD-capable connector names, and NixlPushConnector is not in
# it -- upstream dynamo main has no mention of it either.
#
# WHY THE PULL PROTOCOL IS THE RIGHT ONE, and not a new class. The registry maps
# a connector to how dynamo *orchestrates* the P/D exchange, and the two shipped
# protocols differ in direction:
#
#   NixlConnectorProtocol      "decode-side params come straight off the
#                               engine response"
#   MooncakeConnectorProtocol  transfer_id allocated up front, bootstrap
#                               address published to the decode worker
#
# NixlPushConnector reverses the *data* direction -- P writes into D rather than
# D reading from P -- but not the orchestration. Both nixl schedulers use one
# flag vocabulary: push_scheduler.py decides the role with
# `is_p_node = bool(params.get("do_remote_decode"))` exactly as pull does, reads
# `do_remote_prefill` on the D side, and returns its address on the response
# from `request_finished`. So dynamo's job is identical: send do_remote_decode
# to P, hand P's response params to D verbatim.
#
# That also carries the piece this arm exists for. P puts
# `pp_size=parallel_config.pipeline_parallel_size` into those params
# (push_scheduler.py) and D reads `params.get("pp_size", 1)` back out, so
# pipeline topology crosses the frontend only because the params are passed
# through unaltered -- which is what NixlConnectorProtocol does.
#
# Registering a named subclass rather than aliasing so the log and any traceback
# say which connector is in play.
#
# This is a runtime patch because dynamo is installed from source at job start,
# after setup_script runs; srtctl calls this file from the post-install hook.
# =============================================================================
set -euo pipefail

TARGET=$(python3 - <<'PY'
import importlib.util, os, sys
spec = importlib.util.find_spec("dynamo.vllm.kv_connector_protocols")
if spec is None or not spec.origin:
    sys.exit("dynamo.vllm.kv_connector_protocols not importable")
print(spec.origin)
PY
)
echo "[dynamo-pd] target: ${TARGET}"

python3 - "$TARGET" <<'PY'
import sys

path = sys.argv[1]
src = open(path).read()

if "NixlPushConnectorProtocol" in src:
    print("[dynamo-pd] already patched")
    raise SystemExit(0)

anchor = '''KV_CONNECTOR_PROTOCOLS: Dict[str, Type[KvConnectorProtocol]] = {
    "NixlConnector": NixlConnectorProtocol,'''
if anchor not in src:
    sys.exit(
        "[dynamo-pd] FATAL: KV_CONNECTOR_PROTOCOLS does not have the shape this "
        "patch expects; dynamo has moved and the patch needs revisiting."
    )

klass = '''class NixlPushConnectorProtocol(NixlConnectorProtocol):
    """Push-based NIXL: same orchestration as pull, opposite data direction.

    vLLM's push connector has P write into D rather than D read from P, but the
    two nixl schedulers share one flag vocabulary -- ``do_remote_decode`` marks
    the P node, ``do_remote_prefill`` the D node, and P returns its address on
    the response from ``request_finished``. Dynamo's part is unchanged, so the
    pull protocol's behaviour is correct here; the subclass exists so logs name
    the connector actually in use.
    """


'''
src = src.replace(anchor, klass + anchor, 1)
src = src.replace(
    '    "NixlConnector": NixlConnectorProtocol,\n',
    '    "NixlConnector": NixlConnectorProtocol,\n'
    '    "NixlPushConnector": NixlPushConnectorProtocol,\n',
    1,
)
compile(src, path, "exec")
open(path, "w").write(src)
print("[dynamo-pd] registered NixlPushConnector")
PY

python3 - <<'PY'
import sys
from dynamo.vllm import kv_connector_protocols as k

missing = [n for n in ("NixlConnector", "NixlPushConnector", "MooncakeConnector")
           if n not in k.KV_CONNECTOR_PROTOCOLS]
if missing:
    sys.exit(f"[dynamo-pd] FATAL: still missing from the registry: {missing}")
proto = k.KV_CONNECTOR_PROTOCOLS["NixlPushConnector"]
if not issubclass(proto, k.NixlConnectorProtocol):
    sys.exit("[dynamo-pd] FATAL: NixlPushConnector is not on the pull protocol")
print(f"[dynamo-pd] verified: {sorted(k.KV_CONNECTOR_PROTOCOLS)}")
PY

echo "[dynamo-pd] done"
