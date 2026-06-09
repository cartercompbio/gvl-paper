#! /usr/bin/env python

from pathlib import Path
from typing import Literal

from cyclopts import run

from _pairs import split_pair_batches
from _streaming import drive_loop, prime, run_stream


def bench(
    pairs_parquet: Path,
    pgen: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
    min_seconds: float = 5.0,
    min_batches: int = 10,
):
    import polars as pl
    from genoray import PGEN

    df = pl.read_parquet(pairs_parquet)
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
            res = run_stream(
                gather, batches, min_seconds=min_seconds, min_batches=min_batches
            )
            rows_out.append({
                "dataset": dataset or pgen.name,
                "method": "pgen",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": n_pairs,
                "n_calls": res.distinct_calls,
                "elapsed_ns": res.elapsed_ns,
                "setup_ns": None,
            })
        else:
            from _mem_sampler import PeakRssSampler

            distinct_calls = sum(gather(pairs) for pairs in batches)
            prime(gather, batches)
            with PeakRssSampler() as s:
                drive_loop(
                    gather, batches, min_seconds=min_seconds, min_batches=min_batches
                )
            rows_out.append({
                "dataset": dataset or pgen.name,
                "method": "pgen",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": n_pairs,
                "n_calls": distinct_calls,
                "peak_rss_bytes": s.peak,
            })

    pl.DataFrame(rows_out).write_csv(output)


if __name__ == "__main__":
    run(bench)
