#!/usr/bin/env python3
"""Stop the MLA chunked-prefill workspace from scaling with max_num_seqs.

Port of wzhao18/vllm@2331dddd94 + @d4f1b6438c, which are not upstream: as of
vLLM cb810483 neither the change nor a PR proposing it exists on main.

`determine_chunked_prefill_workspace_size` caps the workspace at 64k tokens and
then floors it at `max_num_seqs * block_size`, on the assumption that a chunk
start has to land on a block boundary. Kimi-K3's block_size is 1536 rather than
32 -- forced up so the attention page covers a KDA/Mamba page -- so the floor
overrides the cap for any max_num_seqs >= 43 and ties the workspace to
max_num_seqs. At TP8 that is ~6.7 KB/token/GPU: 420 MB at the cap against
10.5 GB at max_num_seqs=1024, taken straight out of the KV pool.

The alignment the floor exists for is only needed on the DCP path, whose local
context lengths are padded per block. `gather_and_maybe_dequant_cache` and
`cp_gather_cache` add seq_starts into the absolute token offset before taking
`offset / block_size` and `offset % block_size`, which is exact for any start.
That claim carries no upstream review, so verify generated output against an
unpatched run before trusting any number measured with this applied.

Only `mla_attention.py` is patched. `sparse_mla_attention.py` keeps its own
workspace sizing and its own `align_chunk_to_block=current_platform.is_cuda()`;
that is the DSV4-style sparse path, not K3's.
"""

import argparse
from pathlib import Path

FLOOR = """        # Enforce that we enough for at least 1 page per request
        chunked_prefill_workspace_size = max(
            chunked_prefill_workspace_size,
            scheduler_config.max_num_seqs * cache_config.block_size,
        )"""

PATCHED_FLOOR = """        # Enforce enough workspace that ``workspace // num_prefills_with_context``
        # in build_mla_chunked_context_metadata() stays above zero.
        #
        # DCP still rounds that quotient down to block_size, so it needs a whole
        # page per request or the chunk comes out empty. Everything else only
        # needs one token per request, since the gather kernels handle chunk
        # starts at any offset.
        #
        # Read DCP from the config rather than the process group: this also runs
        # during profiling, before the builder resolves its own dcp_world_size.
        # A config/group mismatch can then only over-allocate, never under.
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            tokens_per_request = cache_config.block_size
        else:
            tokens_per_request = 1

        chunked_prefill_workspace_size = max(
            chunked_prefill_workspace_size,
            scheduler_config.max_num_seqs * tokens_per_request,
        )"""

ALIGN = "                align_chunk_to_block=True,"

PATCHED_ALIGN = """                # DCP pads local context lengths per block, so it still needs
                # block-aligned chunk starts; the gather kernels do not.
                align_chunk_to_block=self.dcp_world_size > 1,"""

MARKER = "tokens_per_request = cache_config.block_size"


def patch_mla(mla_path: Path) -> None:
    source = mla_path.read_text()
    if MARKER in source:
        print(f"MLA chunked-prefill workspace already patched: {mla_path}")
        return

    for old, new in ((FLOOR, PATCHED_FLOOR), (ALIGN, PATCHED_ALIGN)):
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"Expected exactly one patch target in {mla_path}, found {count}: "
                f"{old.strip().splitlines()[0]}"
            )
        source = source.replace(old, new, 1)

    mla_path.write_text(source)
    print(f"Patched MLA chunked-prefill workspace: {mla_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mla-path", type=Path)
    args = parser.parse_args()

    mla_path = args.mla_path
    if mla_path is None:
        import vllm

        mla_path = (
            Path(vllm.__file__).parent
            / "model_executor/layers/attention/mla_attention.py"
        )
    patch_mla(mla_path)


if __name__ == "__main__":
    main()
