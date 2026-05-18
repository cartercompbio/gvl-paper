#! /usr/bin/env python

from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from cyclopts import run


def bench(
    pairs_parquet: Path,
    bcf: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
):
    import numpy as np
    import polars as pl
    from genoray import VCF

    df = pl.read_parquet(pairs_parquet)
    q_len = int(df["end"][0] - df["start"][0])

    rows_out: list[dict] = []

    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        pairs = [
            ((row["contig"], int(row["start"]), int(row["end"])), row["sample"])
            for row in group.iter_rows(named=True)
        ]
        if not pairs:
            continue

        if mode == "throughput":
            _bcf = VCF(bcf, with_gvi_index=False)
            t0 = perf_counter_ns()
            n_calls = 0
            for (contig, start, end), sample in pairs:
                _bcf = _bcf.set_samples(sample)
                genos = _bcf.read(contig, start, end, mode=_bcf.Genos8)
                n_calls += int((genos > 0).sum())
            elapsed_ns = perf_counter_ns() - t0
            rows_out.append({
                "dataset": dataset or bcf.name,
                "method": "bcf",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": len(pairs),
                "n_calls": n_calls,
                "elapsed_ns": elapsed_ns,
                "setup_ns": None,
            })
        else:
            from _mem_sampler import PeakRssSampler

            _bcf = VCF(bcf, with_gvi_index=False)
            n_calls = 0
            with PeakRssSampler() as s:
                for (contig, start, end), sample in pairs:
                    _bcf = _bcf.set_samples(sample)
                    genos = _bcf.read(contig, start, end, mode=_bcf.Genos8)
                    n_calls += int((genos > 0).sum())
            rows_out.append({
                "dataset": dataset or bcf.name,
                "method": "bcf",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": len(pairs),
                "n_calls": n_calls,
                "peak_rss_bytes": s.peak,
            })

    pl.DataFrame(rows_out).write_csv(output)


if __name__ == "__main__":
    run(bench)
