#! /usr/bin/env python
"""Build a gvl 0.6.1 dataset for the memory benchmark.

tcga: variants (bcf) + tracks (bigwigs)  -> serves both haps and tracks
1kgp: variants (pgen) only               -> serves haps
"""
from pathlib import Path

import typer


def main(
    kind: str,          # "tcga" or "1kgp"
    out: Path,
    bed: Path,
    fasta: Path,
    length: int,
    variants: Path,
    bigwig_table: Path = None,  # type: ignore
    max_mem_gb: int = 16,
):
    import genvarloader as gvl

    bigwigs = None
    if kind == "tcga":
        if bigwig_table is None:
            raise ValueError("tcga requires --bigwig-table")
        bigwigs = gvl.BigWigs.from_table("read-depth", str(bigwig_table))

    gvl.write(
        out,
        bed,
        variants=str(variants),
        bigwigs=bigwigs,
        length=length,
        overwrite=True,
        max_mem=max_mem_gb * 2**30,
    )
    print(f"BUILT {out} (kind={kind}, length={length})", flush=True)


if __name__ == "__main__":
    typer.run(main)
