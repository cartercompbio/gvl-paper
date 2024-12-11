#! /usr/bin/env python
from pathlib import Path
from subprocess import CalledProcessError, run

import typer
from loguru import logger

WDIR = Path("/cellar/users/dlaub/projects/GenVarLoader/benchmarking/haplotypes")


def main(
    fasta: Path = Path(
        "/carter/shared/genomes/homo_sapiens/ensembl_grch37.p13_v107/Homo_sapiens.GRCh37.dna.toplevel.fa.bgz"
    ),
    results: Path = WDIR / "ref_results.csv",
    max_nucleotides: int = 2**25,
    bench: bool = True,
):
    from itertools import product

    import numpy as np

    results = results.resolve()

    seqlens = np.array([2048, 16384, 131072, 1048576])

    if bench:
        logger.info("Launching benchmarks.")
        # 1 -> 64
        threads_ls = 2 ** np.arange(0, 7, dtype=np.int64)
        for threads, length in product(threads_ls, seqlens):
            batch_size = 32
            npb = length * batch_size
            n_batches = max(20, -(-max_nucleotides // npb))
            try:
                launch_bench(results, fasta, length, threads, batch_size, n_batches)
            except CalledProcessError as e:
                print(e.stdout, e.stderr)
                raise e


def launch_bench(
    results: Path,
    fasta: Path,
    length: int,
    n_threads: int,
    batch_size: int,
    n_batches: int,
):
    stem = f"reference_threads={n_threads}_seqlen={length}_bs={batch_size}"
    out_file = f"{stem}.out"
    err_file = f"{stem}.err"
    cmd = [
        "sbatch",
        "--partition=carter-compute",
        f"--cpus-per-task={n_threads}",
        f"--output={WDIR / 'out' / out_file}",
        f"--error={WDIR / 'err' / err_file}",
        "--nodelist=carter-cn-04",
        "--mem=16G",
        str(WDIR / "ref_benchmark.py"),
        str(fasta),
        str(length),
        "62",  # n samples
        str(batch_size),
        str(n_batches),
        "--results",
        str(results),
    ]
    run(cmd, check=True)


if __name__ == "__main__":
    typer.run(main)
