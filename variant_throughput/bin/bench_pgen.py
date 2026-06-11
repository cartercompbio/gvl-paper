#! /usr/bin/env python

from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from cyclopts import run

from _pairs import split_pair_batches


def bench(
    pairs_parquet: Path,
    pgen: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
    max_pairs_per_rep: int = 0,
):
    import polars as pl
    from genoray import PGEN

    df = pl.read_parquet(pairs_parquet)
    # Peak RSS is invariant to query count (dominated by the fixed pgen open/index
    # cost), so memory runs can subsample to avoid pathological wall-times on the
    # smallest query lengths (largest batches). 0 = use all pairs.
    if max_pairs_per_rep > 0:
        df = df.group_by("replicate", maintain_order=True).head(max_pairs_per_rep)
    q_len = int(df["end"][0] - df["start"][0])

    rows_out: list[dict] = []

    _pgen = PGEN(pgen)

    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        batches = split_pair_batches(group)
        if not batches:
            continue
        n_pairs = sum(len(pairs) for pairs in batches)

        def gather(pairs) -> int:
            nonlocal _pgen
            n = 0
            for (contig, start, end), sample in pairs:
                _pgen = _pgen.set_samples(sample)
                genos = _pgen.read(contig, start, end, mode=_pgen.Genos)
                n += int((genos > 0).sum())
            return n

        if mode == "throughput":
            t0 = perf_counter_ns()
            n_calls = sum(gather(pairs) for pairs in batches)
            elapsed_ns = perf_counter_ns() - t0
            rows_out.append({
                "dataset": dataset or pgen.name,
                "method": "pgen",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": n_pairs,
                "n_calls": n_calls,
                "elapsed_ns": elapsed_ns,
                "setup_ns": None,
            })
        else:
            from _mem_sampler import PeakRssSampler

            with PeakRssSampler() as s:
                n_calls = sum(gather(pairs) for pairs in batches)
            rows_out.append({
                "dataset": dataset or pgen.name,
                "method": "pgen",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": n_pairs,
                "n_calls": n_calls,
                "peak_rss_bytes": s.peak,
            })

    pl.DataFrame(rows_out).write_csv(output)


if __name__ == "__main__":
    run(bench)
