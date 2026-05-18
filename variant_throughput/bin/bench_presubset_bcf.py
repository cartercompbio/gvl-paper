#! /usr/bin/env python

import os
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from cyclopts import run


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
):
    import polars as pl

    df = pl.read_parquet(pairs_parquet)
    q_len = int(df["end"][0] - df["start"][0])
    tmp_dir = Path(".bench_tmp")

    rows_out: list[dict] = []

    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        pairs = [
            ((row["contig"], int(row["start"]), int(row["end"])), row["sample"])
            for row in group.iter_rows(named=True)
        ]
        if not pairs:
            continue

        tmp_paths: list[str] = []
        try:
            if mode == "throughput":
                t0 = perf_counter_ns()
                tmp_paths = _subset_pairs(bcf, pairs, tmp_dir)
                subset_ns = perf_counter_ns() - t0

                t0 = perf_counter_ns()
                n_calls = _read_subsets(tmp_paths)
                read_ns = perf_counter_ns() - t0

                rows_out.append({
                    "dataset": dataset or bcf.name,
                    "method": "presubset_bcf",
                    "query_length": q_len,
                    "n_samples": int(n_samples),
                    "replicate": int(rep),
                    "n_pairs": len(pairs),
                    "n_calls": n_calls,
                    "elapsed_ns": read_ns,
                    "setup_ns": subset_ns,
                })
            else:
                from _mem_sampler import PeakRssSampler

                tmp_paths = _subset_pairs(bcf, pairs, tmp_dir)
                with PeakRssSampler() as s:
                    n_calls = _read_subsets(tmp_paths)

                rows_out.append({
                    "dataset": dataset or bcf.name,
                    "method": "presubset_bcf",
                    "query_length": q_len,
                    "n_samples": int(n_samples),
                    "replicate": int(rep),
                    "n_pairs": len(pairs),
                    "n_calls": n_calls,
                    "peak_rss_bytes": s.peak,
                })
        finally:
            for path in tmp_paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    pl.DataFrame(rows_out).write_csv(output)


if __name__ == "__main__":
    run(bench)
