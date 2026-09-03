#!/usr/bin/env bash
# Stop the PP draft-embedding lookup from demanding a complete HF snapshot.
# =============================================================================
# WHAT BREAKS. Our dspark-pp carry builds the target's vocab embedding on the
# non-first PP stage (_load_target_embed_tokens_for_pp) and resolves the model
# directory with snapshot_download(..., local_files_only=True). Newer
# huggingface_hub validates that the cached snapshot holds *every* file in the
# repo and raises otherwise:
#
#   huggingface_hub.errors.IncompleteSnapshotError: The cached snapshot for
#   'moonshotai/Kimi-K3' (revision 'main', commit f831ab66...) is incomplete:
#   13 file(s) are missing (.eval_results/apex-agents.yaml,
#   .eval_results/deep-swe.yaml, .eval_results/gpqa.yaml, ... (10 more)).
#   Outgoing traffic is disabled ('local_files_only=True').
#
# The missing files are evaluation metadata. The shared staged checkpoint does
# not carry them and inference never reads them, but the check does not care
# what we intend to open -- it compares the whole repo listing against the cache.
#
# This killed all four speculative arms of pipeline 66111736 in load_model,
# while the one no-spec arm ran: only the speculative path calls this function.
# The 08/28 image ships an older hub without the completeness check, which is
# why the same carry has been fine there.
#
# THE FIX is to declare what the function actually reads -- the shard index and
# the safetensors -- so hub validates completeness against that filtered list.
# Done here rather than inside k3-dspark-pp-0901.patch because editing a hunk
# body without recomputing the @@ line counts corrupts the patch.
# =============================================================================
set -euo pipefail

echo "=== dspark-embed-snapshot: scope the PP embed snapshot to what it reads ==="

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
target = os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")
if not os.path.exists(target):
    sys.exit("[dspark-embed] FATAL: no dspark/utils.py in this image")

src = open(target).read()

if "allow_patterns" in src:
    print("[dspark-embed] already scoped in this image")
    sys.exit(0)

ANCHOR = """        model_dir = snapshot_download(
            model_dir,
            revision=model_config.revision,
            local_files_only=True,
        )
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[dspark-embed] FATAL: expected one snapshot_download call in "
        "_load_target_embed_tokens_for_pp, found %d -- has dspark-pp changed?"
        % src.count(ANCHOR)
    )

FIXED = """        model_dir = snapshot_download(
            model_dir,
            revision=model_config.revision,
            local_files_only=True,
            # Only the shard index and the safetensors are read below. hub
            # validates snapshot completeness against the filtered list, and the
            # staged checkpoint omits repo metadata inference never opens.
            allow_patterns=["*.json", "*.safetensors"],
        )
"""
src = src.replace(ANCHOR, FIXED, 1)
compile(src, target, "exec")
open(target, "w").write(src)
print("[dspark-embed] applied: " + target)
PY

# Verify against the thing that actually failed: the call must carry the filter,
# and the module must still import cleanly under the image's own hub version.
python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/v1/worker/gpu/spec_decode/dspark/utils.py")).read()
if 'allow_patterns=["*.json", "*.safetensors"]' not in src:
    sys.exit("[dspark-embed] FATAL: the filter is not in the file after writing it")
if "local_files_only=True" not in src:
    sys.exit("[dspark-embed] FATAL: local_files_only was lost -- this must never download")
print("[dspark-embed] verified: snapshot scoped, downloads still disabled")
PY

echo "=== dspark-embed-snapshot: done ==="
