#!/usr/bin/env bash
# The B200 DCP8 chain, the column guard, and the probe that says why the column is bad.
# =============================================================================
# The guard reports the result -- 7 of 8 checkpoint columns past a block table of width
# 683 -- and never said what produced it. The arithmetic does not explain it: the
# divisor is MambaSpec.block_size, and max_model_len // that - 1 fits inside 683. So a
# seq_len larger than the model's own maximum is reaching the calculation, and we have
# never printed it. See vllm-container-deps-k3-colprobe.sh.
# =============================================================================
set -euo pipefail

bash /configs/patches/vllm-container-deps-k3-b200-dcp8-diag.sh
bash /configs/patches/vllm-container-deps-k3-ckptcol-bounds.sh
bash /configs/patches/vllm-container-deps-k3-colprobe.sh
