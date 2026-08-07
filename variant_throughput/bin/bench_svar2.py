#! /usr/bin/env python

from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

import numpy as np
from cyclopts import run
from genoray import SparseVar2

from _pairs import split_pair_batches
from _streaming import drive_loop, prime, run_stream

Pair = tuple[tuple[str, int, int], str]


def _svar2_search(sv: SparseVar2, pairs: list[Pair]) -> list[tuple[str, dict]]:
    """Setup phase: the interval search, cached ahead of training by gvl.write.

    Groups pairs by contig and passes the unique sample set, exactly as
    `_svar_search` (SVAR v1) does -- this computes the region x sample
    cross-product for both formats, so the two remain matched.
    """
    by_contig: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    for i, ((c, s, e), smp) in enumerate(pairs):
        by_contig[c].append((i, s, e, smp))

    bundles: list[tuple[str, dict]] = []
    for contig, rows in by_contig.items():
        _, starts_list, ends_list, samples_list = zip(*rows)
        unique_samples = sorted(set(samples_list))
        bundles.append((
            contig,
            sv._find_ranges(
                contig,
                np.asarray(starts_list, dtype=np.int64),
                np.asarray(ends_list, dtype=np.int64),
                samples=np.asarray(unique_samples),
            ),
        ))
    return bundles


def _svar2_gather(sv: SparseVar2, bundles: list[tuple[str, dict]]) -> None:
    """Read phase: tree-free replay. The timed operation for throughput."""
    for contig, bundle in bundles:
        sv._gather_ranges(contig, bundle)


def _svar2_n_calls(sv: SparseVar2, pairs: list[Pair]) -> int:
    """Decode-free call count for one batch, computed OUTSIDE the timed setup.

    `region_counts` returns per-`(region, sample, ploid)` counts over the
    store's FULL cohort on the sample axis, not the selected subset -- so we
    must index out the `(region, sample)` cells this batch's pairs actually
    name before summing. Naively summing the whole array would inflate
    n_calls by the cohort size.
    """
    by_contig: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for (c, s, e), smp in pairs:
        by_contig[c].append((s, e, smp))

    sample_idx = {s: j for j, s in enumerate(sv.available_samples)}

    total = 0
    for contig, rows in by_contig.items():
        starts_list, ends_list, samples_list = zip(*rows)
        # shape (R, S_full, P) -- S_full is the store's full cohort, not the
        # pair list's selected subset.
        counts = sv.region_counts(contig, list(zip(starts_list, ends_list)))
        for r, smp in enumerate(samples_list):
            total += int(counts[r, sample_idx[smp], :].sum())
    return total


def bench(
    pairs_parquet: Path,
    svar2: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
    min_seconds: float = 5.0,
    min_batches: int = 10,
):
    import polars as pl

    sv = SparseVar2(svar2)

    df = pl.read_parquet(pairs_parquet)
    q_len = int(df["end"][0] - df["start"][0])

    rows_out: list[dict] = []

    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        batches = split_pair_batches(group)
        if not batches:
            continue

        # AOT interval search (cached ahead of training); timed once -> setup_ns.
        # n_calls is intentionally NOT computed here -- see _svar2_n_calls.
        t0 = perf_counter_ns()
        bundles_per_batch = [_svar2_search(sv, pairs) for pairs in batches]
        setup_ns = perf_counter_ns() - t0

        # Decode-free call count, computed OUTSIDE the timed setup region.
        payloads = [
            (bundles, _svar2_n_calls(sv, pairs))
            for bundles, pairs in zip(bundles_per_batch, batches)
        ]
        n_pairs = sum(len(pairs) for pairs in batches)

        def gather(p) -> int:
            _svar2_gather(sv, p[0])
            return p[1]

        if mode == "throughput":
            res = run_stream(
                gather, payloads, min_seconds=min_seconds, min_batches=min_batches
            )
            rows_out.append({
                "dataset": dataset or svar2.name,
                "method": "svar2",
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

            distinct_calls = sum(p[1] for p in payloads)
            prime(gather, payloads)
            with PeakRssSampler() as s:
                drive_loop(
                    gather, payloads, min_seconds=min_seconds, min_batches=min_batches
                )
            rows_out.append({
                "dataset": dataset or svar2.name,
                "method": "svar2",
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
