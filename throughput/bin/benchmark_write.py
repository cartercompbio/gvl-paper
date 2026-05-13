#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def _open_variants(variants: Path):
    import genoray

    name = variants.name.lower()
    if name.endswith(".pgen"):
        return genoray.PGEN(variants)
    elif name.endswith(".bcf") or name.endswith(".vcf") or name.endswith(".vcf.gz"):
        return genoray.VCF(variants)
    elif variants.is_dir() or name.endswith(".svar"):
        return genoray.SparseVar(variants)
    else:
        raise ValueError(f"Unknown variant format: {variants}")


def _load_tracks(bigwig_table: Path | None):
    if bigwig_table is None:
        return None
    import polars as pl
    import genvarloader as gvl

    df = pl.read_csv(bigwig_table)
    paths = dict(zip(df["sample"].to_list(), df["path"].to_list()))
    return gvl.BigWigs("read-depth", paths)


def bench(
    results: Path,
    variants: Path,
    bed: Path,
    fasta: Path,
    out_gvl: Path,
    length: int,
    backend: str,
    dataset: str = "",
    bigwig_table: Path | None = None,
    measure_memory: bool = False,
    max_mem: str = "4g",
    max_jitter: int | None = None,
):
    from time import perf_counter_ns

    import genvarloader as gvl

    var_source = _open_variants(variants)
    tracks = _load_tracks(bigwig_table)
    dataset = dataset or variants.stem

    def write():
        gvl.write(
            out_gvl,
            bed,
            variants=var_source,
            tracks=tracks,
            max_jitter=max_jitter,
            overwrite=True,
            max_mem=max_mem,
        )

    if measure_memory:
        from _mem_sampler import PeakRssSampler

        with PeakRssSampler() as s:
            write()
        with open(results, "w") as f:
            f.write("dataset,backend,seqlen,peak_rss_bytes\n")
            f.write(f"{dataset},{backend},{length},{s.peak}\n")
    else:
        t0 = perf_counter_ns()
        write()
        duration = perf_counter_ns() - t0
        with open(results, "w") as f:
            f.write("dataset,backend,seqlen,duration\n")
            f.write(f"{dataset},{backend},{length},{duration}\n")


if __name__ == "__main__":
    run(bench)
