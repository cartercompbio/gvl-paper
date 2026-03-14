#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def bench(
    results: Path,
    ds_path: Path,
    length: int,
    fasta: Path,
    batch_size: int,
    n_batches: int,
    burn_in: int = 5,
    replicates: int = 5,
):
    import os
    from time import perf_counter
    from typing import List

    import genvarloader as gvl
    from filelock import FileLock

    ds = (
        gvl.Dataset.open(ds_path, fasta)
        .with_tracks("read-depth", "tracks")
        .with_len(length)
    )
    dataset = ds_path.parent.name
    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False)

    throughputs: List[float] = []
    for _ in range(replicates):
        n_yielded = 0
        n_nucleotides: int = 0
        t0 = perf_counter()
        while n_yielded < n_batches + burn_in:
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
        throughputs.append(n_nucleotides / seconds / 2**20 * batch.element_size())  # type: ignore

    threads = len(os.sched_getaffinity(0))
    header = "dataset,threads,seqlen,batch_size,throughput\n"
    result = f"{dataset},{threads},{length},{batch_size},{throughputs}\n"
    with FileLock(results.with_suffix(".lock")):
        if results.exists():
            with open(results, "a") as f:
                f.write(result)
        else:
            with open(results, "w") as f:
                f.write(header)
                f.write(result)


if __name__ == "__main__":
    run(bench)
