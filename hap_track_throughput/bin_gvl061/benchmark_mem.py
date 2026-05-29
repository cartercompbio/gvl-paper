#! /usr/bin/env python
"""Average + peak RSS memory benchmark for the GVL 0.6.1 dataloader.

Mirrors the 0.24.1 memory-measurement mode (hap_track_throughput/bin/benchmark_{haps,tracks}.py
--measure-memory) but on the 0.6.1 API, so the manuscript memory figures can be regenerated
on the pinned version. Sweeps the (threads, batch_size) grid, sampling RSS of this process +
descendants while iterating the dataloader, and records avg + peak RSS per cell.

Grid CSV columns: threads,batch_size,n_batches  (as produced by make_launch_grid.py).
"""

from pathlib import Path

import typer


def bench(
    results: Path,
    ds_path: Path,
    fasta: Path,
    grid_file: Path,
    mode: str = "tracks",          # "tracks" or "haps"
    dataset: str = "",
    backend: str = "gvl061",
    burn_in: int = 1,
    replicates: int = 3,
):
    import gc
    import os
    from itertools import product
    from time import sleep

    import genvarloader as gvl
    import numba as nb
    import polars as pl

    from _mem_sampler import PeakRssSampler

    if mode == "tracks":
        ds = gvl.Dataset.open(ds_path, fasta, return_sequences=False)
    elif mode == "haps":
        ds = gvl.Dataset.open(ds_path, fasta, return_tracks=False)
    else:
        raise ValueError(f"mode must be 'tracks' or 'haps', got {mode!r}")

    dataset = dataset or ds_path.parent.name
    length = ds.region_length

    max_threads = len(os.sched_getaffinity(0))
    grid = pl.read_csv(grid_file)
    assert int(grid["threads"].max()) <= max_threads  # type: ignore

    with open(results, "w") as f:
        f.write("dataset,backend,threads,seqlen,batch_size,avg_rss_bytes,peak_rss_bytes\n")
        f.flush()
        for (n_thread, batch_size, n_batches), _ in product(
            grid.iter_rows(), range(replicates)
        ):
            nb.set_num_threads(n_thread)
            dl = ds.to_dataloader(batch_size=batch_size, shuffle=False)
            n_yielded = 0
            done = False
            with PeakRssSampler() as s:
                while not done:
                    for _batch in dl:
                        n_yielded += 1
                        if n_yielded >= n_batches + burn_in:
                            done = True
                            break
            del dl
            gc.collect()
            sleep(0.5)
            f.write(
                f"{dataset},{backend},{n_thread},{length},{batch_size},{s.avg},{s.peak}\n"
            )
            f.flush()  # survive an OOM-kill mid-grid (partial CSV is preserved)


if __name__ == "__main__":
    typer.run(bench)
