#! /usr/bin/env python
"""Haplotype dataloader throughput / memory sweep for the full GVL 0.27.0 bench.

Sweeps a (threads, batch_size, n_batches) grid for one dataset + seqlen, in one
dataloader mode (none|buffered). Throughput mode emits bytes-accounted MiB/s
(directly comparable to results/hap_results.csv); memory mode emits peak/avg RSS.
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
    buffer_bytes: int = 2 * 2**30,  # floor buffer for small cells
    max_buffer_bytes: int = 64 * 2**30,  # per-cell buffer cap; over-cap cells -> NaN
    burn_in: int = 1,
    replicates: int = 3,
    measure_memory: bool = False,
    memory_timeseries: bool = False,  # emit RSS-vs-time growth curve per cell
    growth_time_s: float = 180.0,  # iteration window for the growth curve
    time_limit_s: float = 45.0,
    min_batches: int = 5,
):
    import gc
    import os
    from itertools import product
    from time import perf_counter_ns, sleep

    import genvarloader as gvl
    import numba as nb
    import polars as pl

    from _bench_common import (
        MEMORY_HEADER,
        THROUGHPUT_HEADER,
        measure_cell,
        mib_per_s,
        n_bytes,
    )

    if dl_mode not in ("none", "buffered"):
        raise ValueError(f"dl_mode must be 'none' or 'buffered', got {dl_mode!r}")

    ds = (
        gvl.Dataset.open(ds_path, fasta)
        .with_tracks(False)
        .with_seqs("haplotypes")
        .with_len(length)
        .with_settings(deterministic=True)  # required by buffered haps; harmless for none
    )
    dataset = dataset or ds_path.parent.name

    max_threads = len(os.sched_getaffinity(0))
    grid = pl.read_csv(grid_file)
    assert int(grid["threads"].max()) <= max_threads  # type: ignore

    time_limit_ns = int(time_limit_s * 1e9)

    # Per-cell buffer sizing. The buffered loader is double-buffered, so a
    # mini-batch fits only when batch_bytes <= buffer_bytes / N_SLOTS (see
    # genvarloader/_torch.py: slot_bytes = buffer_bytes // n_slots). We probe the
    # per-instance byte cost once (fixed-length output -> constant per instance)
    # and size each cell's buffer to max(floor, N_SLOTS * batch_bytes * HEADROOM),
    # capped at max_buffer_bytes. Over-cap cells are recorded as NaN, not run.
    N_SLOTS = 2
    HEADROOM = 1.15  # clear the strict `>` plus per-instance offset bytes
    if dl_mode == "buffered":
        _probe = ds.to_dataloader(batch_size=1, shuffle=False)
        bytes_per_instance = n_bytes(next(iter(_probe)))
        del _probe
        gc.collect()
        print(f"bytes_per_instance={bytes_per_instance} (probe bs=1)", flush=True)
    else:
        bytes_per_instance = 0

    def cell_dl_kwargs(batch_size: int):
        """Return to_dataloader kwargs for this cell, or None to skip (over cap)."""
        if dl_mode == "none":
            return {}
        required = int(N_SLOTS * batch_size * bytes_per_instance * HEADROOM)
        if required > max_buffer_bytes:
            return None
        return {"mode": "buffered", "buffer_bytes": max(buffer_bytes, required)}

    def _drive(dl, dur_ns: int) -> None:
        """Iterate the dataloader (re-epoching) for ~dur_ns to fault in pages."""
        t0 = perf_counter_ns()
        while perf_counter_ns() - t0 < dur_ns:
            empty = True
            for _ in dl:
                empty = False
                if perf_counter_ns() - t0 >= dur_ns:
                    break
            if empty:
                break

    if measure_memory and memory_timeseries:
        # RSS growth curve at the (single, largest-batch) operating point.
        from _bench_common import MEMORY_TS_HEADER
        from _mem_sampler import RssTimeSeriesSampler

        growth_ns = int(growth_time_s * 1e9)
        with open(results, "w") as f:
            f.write(MEMORY_TS_HEADER)
            f.flush()
            for n_thread, batch_size, n_batches in grid.iter_rows():
                nb.set_num_threads(n_thread)
                kw = cell_dl_kwargs(batch_size)
                if kw is None:
                    print(f"SKIP growth t={n_thread} bs={batch_size}: buffer > cap", flush=True)
                    continue
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **kw)
                except ValueError as e:
                    print(f"SKIP growth t={n_thread} bs={batch_size}: {e}", flush=True)
                    continue
                prefix = f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size}"
                with RssTimeSeriesSampler(f, prefix, interval_s=0.5) as s:
                    _drive(dl, growth_ns)
                print(f"growth t={n_thread} bs={batch_size}: peak={s.peak} avg={s.avg}", flush=True)
                del dl
                gc.collect()
                sleep(0.5)
    elif measure_memory:
        from _mem_sampler import PeakRssSampler

        with open(results, "w") as f:
            f.write(MEMORY_HEADER)
            f.flush()
            for (n_thread, batch_size, n_batches), _ in product(grid.iter_rows(), range(replicates)):
                nb.set_num_threads(n_thread)
                kw = cell_dl_kwargs(batch_size)
                if kw is None:
                    print(f"SKIP mem t={n_thread} bs={batch_size}: buffer > cap {max_buffer_bytes}", flush=True)
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},nan,nan\n")
                    f.flush()
                    continue
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **kw)
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
                if res is None:  # empty epoch -> NaN
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
                kw = cell_dl_kwargs(batch_size)
                if kw is None:
                    print(f"SKIP cell t={n_thread} bs={batch_size}: buffer > cap {max_buffer_bytes}", flush=True)
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},0,0,0,nan\n")
                    f.flush()
                    continue
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **kw)
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
                if res is None:  # empty epoch -> NaN
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
