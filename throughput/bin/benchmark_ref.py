#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def main(
    fasta: Path,
    length: int,
    n_samples: int,
    batch_size: int,
    n_batches: int,
    results: Path,
    burn_in: int = 5,
    replicates: int = 5,
    measure_memory: bool = False,
):
    import os
    from time import perf_counter
    from typing import List

    import numpy as np
    import polars as pl
    import pysam
    from filelock import FileLock
    from torch.utils.data import DataLoader, Dataset

    threads = len(os.sched_getaffinity(0))
    num_workers = threads - 1

    class Ref(Dataset):
        def __init__(self, path: Path, bed: Path, n_samples: int):
            self.path = path
            self.fasta = None
            self.bed = pl.read_csv(
                bed,
                separator="\t",
                has_header=False,
                new_columns=["contig", "start", "end"],
                schema_overrides={"contig": pl.Utf8},
            )
            self.n_samples = n_samples

            bed_ucsc_convention = self.bed["contig"].str.contains("chr").any()
            with pysam.FastaFile(str(self.path)) as f:
                fasta_ucsc_convention = any(
                    [contig.startswith("chr") for contig in f.references]
                )
            if not bed_ucsc_convention and fasta_ucsc_convention:
                self.bed = self.bed.with_columns("chr" + pl.col("contig"))
            elif bed_ucsc_convention and not fasta_ucsc_convention:
                self.bed = self.bed.with_columns(pl.col("contig").str.slice(3))

        @property
        def shape(self):
            return (self.bed.height, self.n_samples)

        def __len__(self):
            return self.bed.height * self.n_samples

        def __getitem__(self, index: int):
            if self.fasta is None:
                self.fasta = pysam.FastaFile(str(self.path))
            region, sample = map(int, np.unravel_index(index, self.shape))
            contig, start, end = self.bed.row(region)
            seq = np.frombuffer(
                self.fasta.fetch(contig, start, end).encode("ascii").upper(), dtype="S1"
            )
            return seq.view("u1").astype(np.uint8, copy=True)

    bed_dir = Path(__file__).parent / "beds"
    bed = Path(bed_dir / f"tile_{length}.bed")
    ds = Ref(fasta, bed, n_samples)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    if measure_memory:
        from _mem_sampler import PeakRssSampler

        with PeakRssSampler() as s:
            for _ in range(replicates):
                n_yielded = 0
                while n_yielded < n_batches + burn_in:
                    for batch in dl:
                        n_yielded += 1
                        if n_yielded >= n_batches:
                            break

        result = f"FASTA,{threads},{length},{batch_size},{s.peak}"
        with FileLock(results.with_suffix(".lock")):
            if results.exists():
                with open(results, "a") as f:
                    f.write(result + "\n")
            else:
                with open(results, "w") as f:
                    f.write("dataset,threads,seqlen,batch_size,peak_rss_bytes\n")
                    f.write(result + "\n")
    else:
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
            throughputs.append(n_nucleotides / seconds / 2**20)

        result = f"FASTA,{threads},{length},{batch_size},{throughputs}"
        with FileLock(results.with_suffix(".lock")):
            if results.exists():
                with open(results, "a") as f:
                    f.write(result + "\n")
            else:
                with open(results, "w") as f:
                    f.write("dataset,threads,seqlen,batch_size,throughput\n")
                    f.write(result + "\n")


if __name__ == "__main__":
    run(main)
