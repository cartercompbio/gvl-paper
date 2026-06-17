#! /usr/bin/env python
"""Apples-to-apples FASTA / pyBigWig baselines for the GVL 0.27 throughput grid.

Reuses the pysam `Ref` and `gvl.BigWigs` dataset readers but drives them through
`_bench_common.measure_cell` so throughput (MiB/s) is directly comparable to the
GVL numbers in results_gvl027/{haps,tracks}/*_none.csv. Thread counts are swept
by the caller via taskset (one invocation per thread count); num_workers is the
allocated CPU count minus one (the main process), matching a fully-optimized
PyTorch multiprocessing DataLoader (reviewer R2-min1).
"""

from pathlib import Path

from cyclopts import run


def batch_bytes(*, batch_size: int, seqlen: int, bytes_per_bp: int) -> int:
    return batch_size * seqlen * bytes_per_bp


def cell_fits(
    *,
    batch_size: int,
    seqlen: int,
    bytes_per_bp: int,
    num_workers: int,
    prefetch_factor: int,
    mem_cap_bytes: int,
) -> bool:
    """A cell fits if its prefetched batches stay under the RAM cap.

    torch's DataLoader prefetches prefetch_factor batches per worker; the main
    process also holds one. Approximate peak as (num_workers * prefetch_factor + 1)
    decoded batches resident at once.
    """
    resident = (num_workers * prefetch_factor + 1) * batch_bytes(
        batch_size=batch_size, seqlen=seqlen, bytes_per_bp=bytes_per_bp
    )
    return resident <= mem_cap_bytes


def select_cells(grid_file: Path, *, threads: int) -> list[tuple[int, int]]:
    """Distinct (batch_size, n_batches) cells for the given thread count."""
    import polars as pl

    grid = pl.read_csv(grid_file)
    rows = (
        grid.filter(pl.col("threads") == threads)
        .select("batch_size", "n_batches")
        .unique()
        .sort("batch_size")
    )
    return [(int(b), int(n)) for b, n in rows.iter_rows()]
