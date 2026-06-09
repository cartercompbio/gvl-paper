#! /usr/bin/env python

import random
from pathlib import Path

from cyclopts import run

from _pairs import compute_batch_size


def load_gap_intervals(
    contig_map: dict[str, str],
    known_contigs: set[str],
    cache_dir: Path | None = None,
) -> dict[str, list[tuple[int, int]]]:
    import urllib.request

    import polars as pl

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "gvl-paper"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "hg38_gap.txt.gz"
    if not cache_path.exists():
        url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz"
        print(f"Downloading UCSC gap table to {cache_path} ...")
        urllib.request.urlretrieve(url, cache_path)

    df = (
        pl.read_csv(
            cache_path,
            separator="\t",
            has_header=False,
            new_columns=["bin", "chrom", "start", "end", "ix", "n", "size", "type", "bridge"],
        )
        .select(
            pl.col("chrom").replace_strict(contig_map, default=None).alias("contig"),
            "start",
            "end",
        )
        .drop_nulls()
        .filter(pl.col("contig").is_in(list(known_contigs)))
        .sort(["contig", "start"])
    )

    result: dict[str, list[tuple[int, int]]] = {c: [] for c in known_contigs}
    for row in df.iter_rows(named=True):
        result[row["contig"]].append((row["start"], row["end"]))
    return result


def build_allowed_intervals(
    contig_lengths: dict[str, int],
    gap_intervals: dict[str, list[tuple[int, int]]],
) -> list[tuple[str, int, int]]:
    allowed: list[tuple[str, int, int]] = []
    for contig, length in contig_lengths.items():
        gaps = sorted(gap_intervals.get(contig, []))
        pos = 0
        for gap_start, gap_end in gaps:
            if gap_start > pos:
                allowed.append((contig, pos, gap_start))
            pos = max(pos, gap_end)
        if pos < length:
            allowed.append((contig, pos, length))
    return allowed


def sample_region(
    allowed: list[tuple[str, int, int]],
    q_len: int,
    rng: random.Random,
) -> tuple[str, int, int] | None:
    eligible = [(c, s, e) for c, s, e in allowed if e - s >= q_len]
    if not eligible:
        return None
    weights = [e - s - q_len + 1 for _, s, e in eligible]
    (contig, ivl_start, ivl_end) = rng.choices(eligible, weights=weights, k=1)[0]
    start = rng.randint(ivl_start, ivl_end - q_len)
    return contig, start, start + q_len


def bench(
    svar: Path,
    fai: Path,
    query_length: int,
    output: Path,
    seed: int = 0,
    n_replicates: int = 5,
    stream_batches: int = 64,
    bp_budget: int = 2**24,
):
    import polars as pl
    from genoray import SparseVar

    _svar = SparseVar(svar)

    _fai = (
        pl.read_csv(
            fai,
            separator="\t",
            has_header=False,
            new_columns=["contig", "length"],
        )
        .select(
            pl.col("contig").replace_strict(_svar._c_norm.contig_map, default=None),
            "length",
        )
        .drop_nulls()
        .filter(pl.col("contig").is_in(_svar.contigs))
    )
    contig_lengths: dict[str, int] = dict(zip(*_fai.get_columns()))

    gap_intervals = load_gap_intervals(
        contig_map=_svar._c_norm.contig_map,
        known_contigs=set(contig_lengths),
    )
    allowed = build_allowed_intervals(contig_lengths, gap_intervals)

    available_samples = list(_svar.available_samples)
    rng = random.Random(seed)

    batch_size = compute_batch_size(query_length, bp_budget)
    n_pairs = stream_batches * batch_size

    replicates: list[int] = []
    batch_ids: list[int] = []
    contigs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    samples: list[str] = []

    for rep in range(n_replicates):
        for k in range(n_pairs):
            region = sample_region(allowed, query_length, rng)
            if region is None:
                break
            contig, start, end = region
            sample = rng.choice(available_samples)
            replicates.append(rep)
            batch_ids.append(k // batch_size)
            contigs.append(contig)
            starts.append(start)
            ends.append(end)
            samples.append(sample)

    pl.DataFrame(
        {
            "replicate": pl.Series(replicates, dtype=pl.UInt16),
            "batch_id": pl.Series(batch_ids, dtype=pl.UInt32),
            "contig": contigs,
            "start": pl.Series(starts, dtype=pl.Int64),
            "end": pl.Series(ends, dtype=pl.Int64),
            "sample": samples,
        }
    ).write_parquet(output)


if __name__ == "__main__":
    run(bench)
