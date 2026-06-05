#! /usr/bin/env python
"""Timed GVL 0.26.0 dataloader throughput sweep for the parity probe.

Sweeps a (threads, batch_size, n_batches) grid for one dataset + seqlen, in one
output mode (haps|tracks) and one dataloader mode (none|buffered), and records
throughput in MiB/s using the same convention as bin_gvl061 (directly comparable
to results/{hap,track}_results.csv).

Output CSV schema:
    dataset,backend,mode,dl_mode,threads,seqlen,batch_size,throughput (MiB/s)
"""

from pathlib import Path

import cyclopts

from _probe_common import mib_per_s, n_bytes


def bench(
    results: Path,
    ds_path: Path,
    length: int,
    fasta: Path,
    grid_file: Path,
    mode: str = "haps",          # "haps" or "tracks"
    dl_mode: str = "none",       # "none" or "buffered"
    dataset: str = "",
    backend: str = "gvl026",
    buffer_bytes: int = 2 * 2 ** 30,
    burn_in: int = 1,
    replicates: int = 5,
):
    import os
    from time import perf_counter

    import genvarloader as gvl
    import numba as nb
    import polars as pl

    if mode == "haps":
        ds = (
            gvl.Dataset.open(ds_path, fasta)
            .with_seqs("haplotypes")
            .with_tracks(False)
            .with_len(length)
            .with_settings(deterministic=True)  # required by buffered haps
        )
    elif mode == "tracks":
        ds = (
            gvl.Dataset.open(ds_path, fasta)
            .with_seqs(None)
            .with_tracks("read-depth")
            .with_len(length)
        )
    else:
        raise ValueError(f"mode must be 'haps' or 'tracks', got {mode!r}")

    dataset = dataset or ds_path.parent.name
    max_threads = len(os.sched_getaffinity(0))
    grid = pl.read_csv(grid_file)
    assert int(grid["threads"].max()) <= max_threads  # type: ignore

    dl_kwargs = {} if dl_mode == "none" else {"mode": "buffered", "buffer_bytes": buffer_bytes}

    with open(results, "w") as f:
        f.write("dataset,backend,mode,dl_mode,threads,seqlen,batch_size,throughput (MiB/s)\n")
        f.flush()
        for n_thread, batch_size, n_batches in grid.iter_rows():
            nb.set_num_threads(n_thread)
            try:
                dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **dl_kwargs)
            except ValueError as e:
                # buffered: a single mini-batch can exceed buffer_bytes -> construction raises.
                # Record the cell as NaN so the sweep continues and the gap is visible.
                print(f"SKIP cell t={n_thread} bs={batch_size} ({dl_mode}): {e}", flush=True)
                for _ in range(replicates):
                    f.write(f"{dataset},{backend},{mode},{dl_mode},{n_thread},{length},{batch_size},nan\n")
                f.flush()
                continue

            for _ in range(replicates):
                n_yielded = 0
                total_bytes = 0
                t0 = perf_counter()
                done = False
                while not done:
                    for batch in dl:
                        if n_yielded == burn_in:
                            t0 = perf_counter()
                        if n_yielded >= burn_in:
                            total_bytes += n_bytes(batch)
                        n_yielded += 1
                        if n_yielded >= n_batches + burn_in:
                            done = True
                            break
                seconds = perf_counter() - t0
                tput = mib_per_s(total_bytes, seconds) if seconds > 0 else float("nan")
                f.write(f"{dataset},{backend},{mode},{dl_mode},{n_thread},{length},{batch_size},{tput}\n")
                f.flush()
            del dl


if __name__ == "__main__":
    cyclopts.run(bench)
