#! /usr/bin/env python
"""Track dataloader throughput / memory sweep for the full GVL 0.27.0 bench.

Sweeps a (threads, batch_size, n_batches) grid for one dataset + seqlen, in one
dataloader mode (none|buffered). Throughput mode emits bytes-accounted MiB/s
(directly comparable to results/track_results.csv); memory mode emits peak/avg RSS.
"""

from pathlib import Path

from cyclopts import run


def bench(
    results: Path,
    ds_path: Path,
    length: int,
    fasta: Path,
    grid_file: Path,
    dataset: str = "",
    backend: str = "svar",
    dl_mode: str = "none",  # "none" or "buffered"
    buffer_bytes: int = 2 * 2**30,
    burn_in: int = 1,
    replicates: int = 3,
    measure_memory: bool = False,
    time_limit_s: float = 45.0,
    min_batches: int = 5,
):
    import gc
    import os
    from itertools import product
    from time import sleep

    import genvarloader as gvl
    import numba as nb
    import polars as pl

    from _bench_common import (
        MEMORY_HEADER,
        THROUGHPUT_HEADER,
        measure_cell,
        mib_per_s,
    )

    if dl_mode not in ("none", "buffered"):
        raise ValueError(f"dl_mode must be 'none' or 'buffered', got {dl_mode!r}")

    ds = (
        gvl.Dataset.open(ds_path, fasta)
        .with_seqs(None)
        .with_tracks("read-depth", "tracks")
        .with_len(length)
    )
    dataset = dataset or ds_path.parent.name

    max_threads = len(os.sched_getaffinity(0))
    grid = pl.read_csv(grid_file)
    assert int(grid["threads"].max()) <= max_threads  # type: ignore

    time_limit_ns = int(time_limit_s * 1e9)
    dl_kwargs = {} if dl_mode == "none" else {"mode": "buffered", "buffer_bytes": buffer_bytes}

    if measure_memory:
        from _mem_sampler import PeakRssSampler

        with open(results, "w") as f:
            f.write(MEMORY_HEADER)
            f.flush()
            for (n_thread, batch_size, n_batches), _ in product(grid.iter_rows(), range(replicates)):
                nb.set_num_threads(n_thread)
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **dl_kwargs)
                except ValueError as e:
                    print(f"SKIP mem t={n_thread} bs={batch_size} ({dl_mode}): {e}", flush=True)
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},nan,nan\n")
                    f.flush()
                    continue
                with PeakRssSampler() as s:
                    res = measure_cell(
                        dl, burn_in=burn_in, n_batches=n_batches,
                        time_limit_ns=time_limit_ns, min_batches=min_batches,
                    )
                del dl
                gc.collect()
                sleep(0.5)
                if res is None:
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},nan,nan\n")
                else:
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},{s.avg},{s.peak}\n")
                f.flush()
    else:
        with open(results, "w") as f:
            f.write(THROUGHPUT_HEADER)
            f.flush()
            for (n_thread, batch_size, n_batches), _ in product(grid.iter_rows(), range(replicates)):
                nb.set_num_threads(n_thread)
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **dl_kwargs)
                except ValueError as e:
                    print(f"SKIP cell t={n_thread} bs={batch_size} ({dl_mode}): {e}", flush=True)
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},0,0,0,nan\n")
                    f.flush()
                    continue
                res = measure_cell(
                    dl, burn_in=burn_in, n_batches=n_batches,
                    time_limit_ns=time_limit_ns, min_batches=min_batches,
                )
                del dl
                gc.collect()
                if res is None:
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},0,0,0,nan\n")
                else:
                    tput = (
                        mib_per_s(res.total_bytes, res.duration_ns / 1e9)
                        if res.duration_ns > 0
                        else float("nan")
                    )
                    f.write(
                        f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},"
                        f"{res.n_measured},{res.total_bytes},{res.duration_ns},{tput}\n"
                    )
                f.flush()


if __name__ == "__main__":
    run(bench)
