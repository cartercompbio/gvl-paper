#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def bench(
    results: Path,
    ds_path: Path,
    length: int,
    fasta: Path,
    grid_file: Path,
    burn_in: int = 5,
    replicates: int = 5,
):
    import os
    from itertools import product
    from time import perf_counter_ns

    import genvarloader as gvl
    import numba as nb
    import polars as pl

    ds = (
        gvl.Dataset.open(ds_path, fasta)
        .with_tracks("read-depth", "tracks")
        .with_len(length)
    )
    dataset = ds_path.parent.name

    max_threads = len(os.sched_getaffinity(0))
    grid = pl.read_csv(grid_file)
    assert int(grid["threads"].max()) <= max_threads  # type: ignore

    with open(results, "w") as f:
        f.write("dataset,threads,seqlen,batch_size,duration\n")
        for (n_thread, batch_size, n_batches), _ in product(
            grid.iter_rows(), range(replicates)
        ):
            nb.set_num_threads(n_thread)
            dl = ds.to_dataloader(batch_size=batch_size, shuffle=False)
            n_yielded = 0
            t0 = perf_counter_ns()
            while n_yielded < n_batches + burn_in:
                for batch in dl:
                    if n_yielded == burn_in:
                        t0 = perf_counter_ns()
                    n_yielded += 1
                    if n_yielded >= n_batches:
                        break
                    pass
            duration = perf_counter_ns() - t0
            f.write(f"{dataset},{n_thread},{length},{batch_size},{duration}\n")


if __name__ == "__main__":
    run(bench)
