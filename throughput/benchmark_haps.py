#! /usr/bin/env python

from pathlib import Path
from typing import List

import torch
import typer


def bench(
    results: Path,
    ds_path: Path,
    fasta: Path,
    batch_size: int,
    n_batches: int,
    burn_in: int = 5,
    replicates: int = 5,
):
    import os
    from time import perf_counter

    import genvarloader as gvl
    from filelock import FileLock

    ds = gvl.Dataset.open(ds_path, fasta, return_tracks=False)
    dataset = ds_path.parent.name
    length = ds.region_length
    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False)

    throughputs: List[float] = []
    for _ in range(replicates):
        n_yielded = 0
        n_nucleotides: int = 0
        t0 = perf_counter()
        while n_yielded < n_batches + burn_in:
            batch: torch.Tensor
            for batch in dl:
                if n_yielded == burn_in:
                    t0 = perf_counter()
                if n_yielded >= burn_in:
                    n_nucleotides += batch.numel()
                n_yielded += 1
                if n_yielded >= n_batches:
                    break
                pass
        seconds = perf_counter() - t0
        throughputs.append(n_nucleotides / seconds / 2**20)

    threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    with FileLock(results.with_suffix(".csv.lock"), timeout=10):
        with open(results, "a") as f:
            if f.tell() == 0:
                f.write("dataset,threads,seqlen,batch_size,throughput (MiB/s)\n")
            for throughput in throughputs:
                f.write(f"{dataset},{threads},{length},{batch_size},{throughput}\n")


if __name__ == "__main__":
    typer.run(bench)
