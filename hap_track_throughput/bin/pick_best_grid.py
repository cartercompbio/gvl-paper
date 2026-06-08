#!/usr/bin/env python3
"""Derive best-throughput operating-point grids for the memory pass.

Reads throughput result CSVs emitted by benchmark_haps.py / benchmark_tracks.py
(schema: dataset,backend,dl_mode,threads,seqlen,batch_size,n_batches_measured,
total_bytes,duration_ns,throughput (MiB/s)), and for each (dataset, seqlen)
writes a ONE-ROW launch grid (threads,batch_size,n_batches) at the cell with the
highest mean throughput. The memory pass consumes these instead of sweeping.

Files are written as ``<out>/<dataset>_<seqlen>_<kind>.csv`` so benchmark.nf can
look them up by params.dataset + seqlen.
"""

from __future__ import annotations

from pathlib import Path

import cyclopts


def ceil_idiv(a: int, b: int) -> int:
    return -(-a // b)


def main(
    haps_dir: Path,
    tracks_dir: Path,
    out_dir: Path,
):
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for kind, d in (("haps", haps_dir), ("tracks", tracks_dir)):
        if not d.exists():
            continue
        csvs = sorted(d.glob("*.csv"))
        if not csvs:
            continue
        df = pl.concat([pl.read_csv(c) for c in csvs], how="vertical_relaxed")
        # The throughput column reads as str when any cell is the literal "nan"
        # (e.g. tracks cells over the buffer cap), so cast before comparing.
        df = df.with_columns(
            pl.col("throughput (MiB/s)").cast(pl.Float64, strict=False)
        )
        # keep only valid measurements
        df = df.filter(
            pl.col("throughput (MiB/s)").is_finite()
            & (pl.col("throughput (MiB/s)") > 0)
        )
        if df.is_empty():
            continue
        # mean throughput per cell, then argmax per (dataset, seqlen)
        agg = df.group_by("dataset", "seqlen", "threads", "batch_size").agg(
            pl.col("throughput (MiB/s)").mean().alias("tput")
        )
        best = (
            agg.sort("tput", descending=True)
            .group_by("dataset", "seqlen", maintain_order=True)
            .first()
        )
        for row in best.iter_rows(named=True):
            dataset, seqlen = row["dataset"], int(row["seqlen"])
            t, bs = int(row["threads"]), int(row["batch_size"])
            n_batches = max(10, ceil_idiv(2**29, max(1, seqlen * bs)))
            out = out_dir / f"{dataset}_{seqlen}_{kind}.csv"
            pl.DataFrame(
                {"threads": [t], "batch_size": [bs], "n_batches": [n_batches]}
            ).write_csv(out)
            print(
                f"{out.name}: threads={t} batch_size={bs} "
                f"n_batches={n_batches} (tput={row['tput']:.1f} MiB/s)"
            )
            n_written += 1
    print(f"wrote {n_written} best-setting grid(s) to {out_dir}")


if __name__ == "__main__":
    cyclopts.run(main)
