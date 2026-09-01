#!/usr/bin/env bash
# ARM 2: arm 1, plus Wei's compact group I/O and load sub-batching.
# =============================================================================
# Everything arm 1 has, plus three commits from github.com/wzhao18/vllm branch
# wzhao/k3-nvfp4-perf, and the two config flags that turn them on. Read arm 1's
# header first: this stack only makes sense on #53324.
#
# WHAT THE FLAGS ARE FOR. Wei's own accounting of a DCP8+DEP16 AgentX run --
# which loses throughput and external-prefix-cache effectiveness at c80, the
# same shape as ours -- models the Mooncake population to within 0.27% of the
# observed 3,226.86 GiB and finds two amplifiers. PMU tail storage keeps every
# historical turn-boundary checkpoint: 88.9% of live keys and 8.03x the normal
# path's bytes. And every value is serialized as a full shared physical page, so
# a KDA group with 23 real layers writes a 29-segment page; 20.69% of
# transferred bytes are padding. DSpark widens that shared page from 24 to 29
# slots, adding 20.83% to every stored value, of which only 17.23 GiB is actual
# draft state and about 540.66 GiB is KDA values dragged along.
#
# compact_group_io writes group-specific regions instead of the whole page. His
# measurement: c80 terminal occupancy 3.23 TB with eviction -> 1.98 TB without,
# external-prefix hit 92.7%, GSM8K 0.944; c120 still zero eviction at 90.2%.
#
# max_load_batch_keys splits one BatchGet into sub-batches. Half of that commit
# is a knob and half is a bug fix: after a partial failure it now invalidates
# the blocks of every sub-batch it never attempted, which previously stayed
# marked loaded with nothing written into them.
#
# BOTH FLAGS DEFAULT OFF in the code. This script only makes them available;
# the config is what turns them on, which is why the yml carries them and not
# this file.
#
# WHY WE EXPECT THIS TO MATTER HERE AND NOT ONLY ON oci-aga. Wei suspects the
# load timeout is specific to a single-RDMA-device cluster. Our reason is
# independent of his being right: our Mooncake external hit rate is the standing
# gap against the AMD curve -- LMCache 85.5% against our 9.4% on the run that
# lost -- and both flags act on that path.
set -euo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"

bash "$HERE/kimi-k3-nightly-v8-mc53324.sh"
bash /configs/patches/vllm-container-deps-k3-wei-mooncake-io.sh
echo "=== v8-weimc ready: v8 (no mooncake) + #53324 + wei compact I/O ==="
