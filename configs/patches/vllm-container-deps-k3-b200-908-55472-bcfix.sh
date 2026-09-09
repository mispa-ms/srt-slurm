#!/usr/bin/env bash
# The 09/08 chain with upstream #55472 plus the PP draft-broadcast race fix.
# = vllm-container-deps-k3-b200-908-55472.sh + vllm-container-deps-k3-pp-broadcast-drafts-fix.sh
# If this arm serves where -908-55472 died (PP1 gather assert a few requests
# in), the race in pp_utils.broadcast_drafts is the third 09/08 wall.
set -euo pipefail
echo "=== k3-b200-908-55472-bcfix ==="
bash /configs/patches/vllm-container-deps-k3-b200-908-55472.sh
bash /configs/patches/vllm-container-deps-k3-pp-broadcast-drafts-fix.sh
python3 - <<'PY'
import importlib.util, os, sys
root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
pu = open(os.path.join(root, "vllm/v1/worker/gpu/pp_utils.py")).read()
i = pu.index("def broadcast_drafts"); j = pu.index("def receive", i)
body = pu[i:j]
if body.index("send = draft_tokens[input_batch.idx_mapping]") > body.index("with torch.cuda.stream(self.broadcast_stream)"):
    sys.exit("[908-55472-bcfix] FATAL: gather still inside the side-stream block")
print("[908-55472-bcfix] verified: gather precedes the side-stream block")
PY
echo "=== k3-b200-908-55472-bcfix: done ==="
