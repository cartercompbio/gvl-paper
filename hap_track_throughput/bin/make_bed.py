#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def make_bed(
    length: int,
    fasta: Path,
    output: Path,
    canonical: bool = True,
    n_samples: int | None = None,
    region: str | None = None,
    seed: int | None = 42,
):
    import re
    from typing import cast

    import polars as pl
    import polars.selectors as cs
    import pyranges1 as pr
    from natsort import natsorted

    bed = pl.read_csv(
        f"{fasta}.fai",
        separator="\t",
        has_header=False,
        columns=range(2),
        new_columns=["Chromosome", "End"],
        schema_overrides={"Chromosome": pl.Utf8, "End": pl.Int32},
    )
    chroms = natsorted(bed["Chromosome"].unique())

    if canonical:
        bed = bed.filter(
            pl.col("Chromosome").str.contains(r"^(chr)?(\d{1,2}|X|Y|MT|M)$")
        )

    region_start = 0
    if region is not None:
        m = re.fullmatch(r"([^:]+)(?::(\d+)-(\d+))?", region)
        if m is None:
            raise ValueError(f"Could not parse region {region!r}; expected 'chrom' or 'chrom:start-end'.")
        r_chrom, r_start, r_end = m.group(1), m.group(2), m.group(3)
        bed = bed.filter(pl.col("Chromosome") == r_chrom)
        if bed.height == 0:
            raise ValueError(f"Region chromosome {r_chrom!r} not present in {fasta}.fai.")
        if r_start is not None and r_end is not None:
            region_start = int(r_start)
            bed = bed.with_columns(End=pl.col("End").clip(upper_bound=int(r_end)))

    bed = pr.PyRanges(bed.with_columns(Start=region_start).to_pandas()).tile_ranges(length)
    bed = cast(pl.DataFrame, pl.from_pandas(bed))
    assert ((bed["End"] - bed["Start"]) == length).all()

    if n_samples is not None:
        bed = (
            bed.group_by("Chromosome")
            .agg(pl.all().shuffle(seed).head(n_samples))
            .explode(cs.exclude("Chromosome"))
        )

    (
        bed.sort(pl.col("Chromosome").cast(pl.Enum(chroms)), "Start")
        .select("Chromosome", "Start", "End")
        .write_csv(output, separator="\t", include_header=False)
    )


if __name__ == "__main__":
    run(make_bed)
