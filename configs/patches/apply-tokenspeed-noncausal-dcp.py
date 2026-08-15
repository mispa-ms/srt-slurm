#!/usr/bin/env python3
"""Declare that TokenspeedMLA supports non-causal multi-token decode under DCP.

    python3 configs/patches/apply-tokenspeed-noncausal-dcp.py <site-packages>

One line, and it has to be set explicitly because the check that reads it
cannot tell "unsupported" from "not declared":

    @classmethod
    def supports_non_causal_dcp(cls) -> bool:
        builder_cls = cls.get_builder_cls()
        return bool(getattr(builder_cls, "supports_non_causal_multi_token_dcp", False))

`getattr(..., False)`. A missing attribute is silently a "no", and the backend
selector then rejects TOKENSPEED_MLA with

    ValueError: Selected backend AttentionBackendEnum.TOKENSPEED_MLA is not
    valid for this configuration.
    Reason: ['non-causal MLA attention with DCP not supported']

every worker fails to start, and the job dies twenty minutes in as an EOFError
from the executor -- three layers away from the missing line.

It arrives with Wei's c147f55594 as a hunk in tokenspeed_mla.py, and that hunk
is the one hunk of the v3 port that did not apply: our de19be2460 moved the
class body, so the context no longer matched. It was then classified as
"already present" on the strength of a neighbouring attribute,
`supports_mtp_with_cp_non_trivial_interleave_size`, which does sit a few lines
away and is a different flag entirely. Two similar names, one read too quickly,
and a ladder lost.

So this asserts the name it sets, rather than trusting that a similar one
nearby means the same thing.
"""

import pathlib
import sys

TARGET = "vllm/v1/attention/backends/mla/tokenspeed_mla.py"
FLAG = "supports_non_causal_multi_token_dcp"

# The sibling flag on the builder, present since Wei's earlier work. It anchors
# the insertion and proves we are in the builder class, not the impl class --
# they both carry `supports_*` attributes and only one of them is read here.
ANCHOR = "    supports_non_causal_multi_token_decode: ClassVar[bool] = True\n"
ADD = "    supports_non_causal_multi_token_dcp: ClassVar[bool] = True\n"


def main() -> int:
    site = pathlib.Path(sys.argv[1])
    path = site / TARGET
    if not path.is_file():
        print(f"[tokenspeed-dcp] FATAL: {path} not found", file=sys.stderr)
        return 1

    src = path.read_text()
    if FLAG in src:
        print("[tokenspeed-dcp] already present")
        return 0
    if src.count(ANCHOR) != 1:
        print(
            f"[tokenspeed-dcp] FATAL: expected exactly one "
            f"supports_non_causal_multi_token_decode declaration, found "
            f"{src.count(ANCHOR)}. The builder class changed shape; read it "
            f"before assuming where this belongs.",
            file=sys.stderr,
        )
        return 1

    path.write_text(src.replace(ANCHOR, ANCHOR + ADD, 1))
    print(f"[tokenspeed-dcp] applied: {FLAG} = True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
