#! /usr/bin/env python
"""Convert a subset PGEN to an SVAR2 store, paralleling BUILD_SVAR_FROM_PGEN's
`genoray write` (v1). The subset PGEN is already normalized (derived from
1kGP.snp_indel.split_multiallelics), so this trusts it via `no_reference=True`
exactly as the v1 build path does.
"""

from pathlib import Path

from cyclopts import run
from genoray import SparseVar2


def build(pgen: Path, out: Path, threads: int = 1):
    SparseVar2.from_pgen(out, pgen, no_reference=True, threads=threads, overwrite=True)


if __name__ == "__main__":
    run(build)
