#!/usr/bin/env bash
# Log the inputs that produce checkpoint_cols, not just the result.
# =============================================================================
# WHY. An earlier probe reported the *result* -- "7 of 8 checkpoint columns out of
# range for a block table of width 683" -- and nothing that produced it. Backward
# tracing then stalled on arithmetic that does not add up:
#
#   checkpoint_cols = seq_len // block_size - 1        (kda_metadata.py:600)
#
#   block_size is MambaSpec.block_size, asserted at line 364, so it is the mamba
#   block size and not the scheduler's -- that class of bug is excluded here.
#   With max_model_len 1,048,576 and a block table 683 wide, the implied block size
#   is ~1,536, and 1,048,576 // 1,536 - 1 = 681, which fits.
#
# So for a column to exceed 683 the seq_len feeding it must be larger than the model's
# own maximum. That is not a bounds bug; it is a bad input, and we have never printed
# it.
#
# There is a plausible source. seq_lens comes from
#
#   seq_lens = m.seq_lens_cpu_upper_bound.tolist()      (kda_metadata.py:584)
#   seq_len  = seq_lens[row]                            (kda_metadata.py:589)
#
# taken whole, while the same buffer is sliced to [:num_reqs] everywhere else --
# mamba_hybrid.py:241 does exactly that two lines before handing it over unsliced at
# :291. If the buffer is longer than the live request count, rows past the end read
# whatever was last there.
#
# THIS ARM MEASURES AND FIXES NOTHING. Seven fixes have already been written against
# this fault on guesswork and all seven failed; the only two facts that survived came
# from instrumentation. For each row whose column lands outside the block table it
# prints the five numbers that fully determine which of three things is wrong:
#
#   seq_len > max_model_len            -> a bad value reached the buffer (look upstream)
#   block_size far below the mamba one -> the wrong spec after all
#   both sane but the column too large -> the width we compare against is the wrong
#                                         group's block table, i.e. the earlier probe
#                                         was measuring the wrong thing
#
# Chained after the column guard so the run still survives; the guard's own report is
# the result, this is the cause.
# =============================================================================
set -euo pipefail

echo "=== colprobe: log the inputs behind checkpoint_cols ==="

python3 - <<'PY'
import importlib.util
import os
import sys

target = os.path.join(
    os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)),
    "vllm/models/kimi_k3/nvidia/kda_metadata.py",
)
src = open(target).read()

if "[colprobe]" in src:
    print("[colprobe] already applied: " + target)
    sys.exit(0)

ANCHOR = """                checkpoint_splits.append((first_len, query_len - first_len))
                checkpoint_cols.append(seq_len // block_size - 1 if valid else -1)
"""
if src.count(ANCHOR) != 1:
    sys.exit(
        "[colprobe] FATAL: expected one checkpoint column append, found %d"
        % src.count(ANCHOR)
    )

ADDITION = '''                checkpoint_splits.append((first_len, query_len - first_len))
                _cp_col = seq_len // block_size - 1 if valid else -1
                # [colprobe] Diagnostic only. Print the inputs when the column cannot
                # index this request's block-table row.
                _cp_w = m.block_table_tensor.size(1)
                if _cp_col >= _cp_w and _COLPROBE["n"] < 40:
                    _COLPROBE["n"] += 1
                    logger.warning(
                        "[colprobe] #%d row=%d col=%d >= width=%d | seq_len=%d "
                        "query_len=%d block_size=%d valid=%s | max_model_len=%s | "
                        "num_reqs=%d len(seq_lens)=%d block_table=%s",
                        _COLPROBE["n"],
                        row,
                        _cp_col,
                        _cp_w,
                        seq_len,
                        query_len,
                        block_size,
                        valid,
                        getattr(
                            getattr(self, "vllm_config", None),
                            "model_config",
                            None,
                        )
                        and self.vllm_config.model_config.max_model_len,
                        m.num_reqs,
                        len(seq_lens),
                        tuple(m.block_table_tensor.shape),
                    )
                checkpoint_cols.append(_cp_col)
'''

src = src.replace(ANCHOR, ADDITION, 1)

# Module-level counter, and a logger if the file lacks one (it does).
if "init_logger" not in src:
    IMPORT_ANCHOR = "from vllm.config import VllmConfig\n"
    if src.count(IMPORT_ANCHOR) != 1:
        sys.exit(
            "[colprobe] FATAL: expected one VllmConfig import to anchor the logger to,"
            " found %d" % src.count(IMPORT_ANCHOR)
        )
    src = src.replace(
        IMPORT_ANCHOR, IMPORT_ANCHOR + "from vllm.logger import init_logger\n", 1
    )

DEF_ANCHOR = "\n@dataclass\n"
if DEF_ANCHOR not in src:
    sys.exit("[colprobe] FATAL: no dataclass to anchor the module state to")
if "logger = init_logger(__name__)" not in src:
    src = src.replace(DEF_ANCHOR, "\nlogger = init_logger(__name__)\n" + DEF_ANCHOR, 1)
src = src.replace(
    DEF_ANCHOR, '\n# [colprobe] report cap\n_COLPROBE = {"n": 0}\n' + DEF_ANCHOR, 1
)

compile(src, target, "exec")
open(target, "w").write(src)
print("[colprobe] applied: " + target)
PY

python3 - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
src = open(os.path.join(root, "vllm/models/kimi_k3/nvidia/kda_metadata.py")).read()
if src.count("[colprobe]") < 3:
    sys.exit("[colprobe] FATAL: markers missing after write")
if "checkpoint_cols.append(_cp_col)" not in src:
    sys.exit("[colprobe] FATAL: the column is no longer appended")

import vllm.models.kimi_k3.nvidia.kda_metadata as km

for name in ("logger", "_COLPROBE"):
    if name not in dir(km):
        sys.exit("[colprobe] FATAL: %s missing from the module" % name)
print("[colprobe] verified: module imports, column still appended, logger present")
PY

echo "=== colprobe: done ==="
