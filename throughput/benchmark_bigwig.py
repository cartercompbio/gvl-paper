#! /usr/bin/env python

from pathlib import Path

import typer

WDIR = Path(__file__).parent


def main(
    length: int,
    batch_size: int,
    n_batches: int,
    bigwig_table: Path = Path(
        "/carter/shared/data/ml4gland/tcga-atac/data/sample_to_bigwig.csv"
    ),
    results: Path = WDIR / "pybigwig_results.csv",
    burn_in: int = 5,
    replicates: int = 5,
):
    import os
    from time import perf_counter
    from typing import List

    import genvarloader as gvl
    import numpy as np
    import polars as pl
    from filelock import FileLock
    from torch.utils.data import DataLoader, Dataset

    threads = len(os.sched_getaffinity(0))
    num_workers = threads - 1

    class BigWigDataset(Dataset):
        def __init__(self, bigwigs: gvl.BigWigs, bed: Path):
            self.bigwigs = bigwigs
            self.bed = pl.read_csv(
                bed,
                separator="\t",
                has_header=False,
                new_columns=["contig", "start", "end"],
                schema_overrides={"contig": pl.Utf8},
            )
            self.n_samples = len(self.bigwigs.samples)

        @property
        def shape(self):
            return (self.bed.height, self.n_samples)

        def __len__(self):
            return self.bed.height * self.n_samples

        def __getitem__(self, index: int):
            region, sample = map(int, np.unravel_index(index, self.shape))
            contig, start, end = self.bed.row(region)
            track = self.bigwigs.read(
                contig, start, end, sample=self.bigwigs.samples[sample]
            )
            return track

    bed_dir = Path(__file__).parent / "beds"
    bed = Path(bed_dir / f"tile_{length}.bed")
    bigwigs = gvl.BigWigs.from_table("bw", bigwig_table)
    ds = BigWigDataset(bigwigs, bed)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

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

    with FileLock(results.with_suffix(".csv.lock"), timeout=10):
        with open(results, "a") as f:
            if f.tell() == 0:
                f.write("dataset,threads,seqlen,batch_size,throughput (MiB/s)\n")
            for throughput in throughputs:
                f.write(f"pybigwig,{threads},{length},{batch_size},{throughput}\n")


if __name__ == "__main__":
    typer.run(main)
