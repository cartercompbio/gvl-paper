#! /usr/bin/env python

import os
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from cyclopts import run

from _pairs import split_pair_batches
from _streaming import drive_loop, prime, run_stream


def _subset_pairs(
    bcf: Path,
    pairs: list[tuple[tuple[str, int, int], str]],
    tmp_dir: Path,
) -> list[str]:
    """Run bcftools view to pre-subset each pair into a temp BCF. Returns tmp paths."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths: list[str] = []
    for (contig, start, end), sample in pairs:
        fd, path = tempfile.mkstemp(suffix=".bcf", dir=tmp_dir)
        os.close(fd)
        tmp_paths.append(path)
        subprocess.run(
            [
                "bcftools",
                "view",
                "-s",
                sample,
                "-r",
                f"chr{contig}:{start + 1}-{end}",
                "--min-ac",
                "1",
                "--no-update",
                "-Ob",
                "-o",
                path,
                str(bcf),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return tmp_paths


def _read_subsets(tmp_paths: list[str]) -> int:
    """Read pre-subsetted BCFs with cyvcf2. Returns n_calls."""
    import numpy as np
    import cyvcf2

    n_calls = 0
    for path in tmp_paths:
        vcf = cyvcf2.VCF(path)
        chunks = []
        for v in vcf:
            chunks.append(v.genotype.array())
        if chunks:
            n_calls += int((np.concatenate(chunks, axis=0)[:, :2] > 0).sum())
        vcf.close()
    return n_calls


def bench(
    pairs_parquet: Path,
    bcf: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
    min_seconds: float = 5.0,
    min_batches: int = 10,
):
    import polars as pl

    df = pl.read_parquet(pairs_parquet)
    q_len = int(df["end"][0] - df["start"][0])
    tmp_dir = Path(".bench_tmp")

    rows_out: list[dict] = []

    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        batches = split_pair_batches(group)
        if not batches:
            continue
        n_pairs = sum(len(pairs) for pairs in batches)

        # AOT: pre-subset each batch into temp BCFs (cached); timed once -> setup_ns.
        t0 = perf_counter_ns()
        batch_tmp = [_subset_pairs(bcf, pairs, tmp_dir) for pairs in batches]
        setup_ns = perf_counter_ns() - t0

        def gather(tmp_paths) -> int:
            return _read_subsets(tmp_paths)

        try:
            if mode == "throughput":
                res = run_stream(
                    gather, batch_tmp, min_seconds=min_seconds, min_batches=min_batches
                )
                rows_out.append({
                    "dataset": dataset or bcf.name,
                    "method": "presubset_bcf",
                    "query_length": q_len,
                    "n_samples": int(n_samples),
                    "replicate": int(rep),
                    "n_pairs": n_pairs,
                    "n_calls": res.distinct_calls,
                    "elapsed_ns": res.elapsed_ns,
                    "setup_ns": setup_ns,
                })
            else:
                from _mem_sampler import PeakRssSampler

                distinct_calls = sum(gather(tp) for tp in batch_tmp)
                prime(gather, batch_tmp)
                with PeakRssSampler() as s:
                    drive_loop(
                        gather, batch_tmp, min_seconds=min_seconds, min_batches=min_batches
                    )
                rows_out.append({
                    "dataset": dataset or bcf.name,
                    "method": "presubset_bcf",
                    "query_length": q_len,
                    "n_samples": int(n_samples),
                    "replicate": int(rep),
                    "n_pairs": n_pairs,
                    "n_calls": distinct_calls,
                    "peak_rss_bytes": s.peak,
                })
        finally:
            for tmp_paths in batch_tmp:
                for path in tmp_paths:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass

    pl.DataFrame(rows_out).write_csv(output)


if __name__ == "__main__":
    run(bench)
