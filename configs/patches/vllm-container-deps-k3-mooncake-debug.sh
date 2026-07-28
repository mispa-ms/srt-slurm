#!/usr/bin/env bash
# Kimi-K3 + Mooncake, with the store connector's own loggers raised to DEBUG.
# =============================================================================
# WHY: on K3 the Mooncake store registers its 1.46 TB of capacity and the master
#   stays up for the whole hour, but the master's admin metrics report
#   `Keys: 0` and `PutStart:(Req=0/0)` -- the connector never issues a put, and
#   `External prefix cache hit rate` is 0.0% for every run as a result. The
#   scheduler and worker log the save/load decision per request, but only at
#   DEBUG:
#     scheduler.py  "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, ..."
#     data.py       "request:%s, meta save spec:%s, meta load spec:%s"
#   so this run is what tells us whether ReqMeta ever carries a save at all.
#
# A blanket VLLM_LOGGING_LEVEL=DEBUG floods the log from eight ranks. vLLM's
# VLLM_LOGGING_CONFIG_PATH replaces the whole dictConfig (logger.py:193), so the
# config below keeps `vllm` at INFO and raises only the four mooncake store
# modules. Child loggers inherit, and propagate=False on `vllm` keeps records off
# the root handler.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-mooncake.sh

LOG_CFG="${VLLM_LOGGING_CONFIG_PATH:-/tmp/vllm_mooncake_debug_logging.json}"
MC="vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store"

python3 - "$LOG_CFG" "$MC" <<'PY'
import json, sys

path, mc = sys.argv[1], sys.argv[2]
fmt = "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s"
cfg = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"vllm": {"class": "vllm.logging_utils.NewLineFormatter",
                            "datefmt": "%m-%d %H:%M:%S", "format": fmt}},
    "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm",
                          "level": "DEBUG", "stream": "ext://sys.stdout"}},
    "loggers": {
        "vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": False},
        **{f"{mc}.{m}": {"level": "DEBUG", "propagate": True}
           for m in ("scheduler", "worker", "data", "connector", "coordinator")},
    },
}
with open(path, "w") as f:
    json.dump(cfg, f, indent=1)
print(f"[mooncake-debug] wrote {path}")
PY

# NOTE: the worker preamble runs this with `bash <script>` (worker_stage.py:79), a
# subshell, so an export here would never reach `vllm serve`. The config must set
# VLLM_LOGGING_CONFIG_PATH in backend.aggregated_environment to the same path.
echo "=== k3-mooncake-debug: wrote ${LOG_CFG}; the config must point"
echo "    VLLM_LOGGING_CONFIG_PATH at it (an export here would not propagate) ==="
