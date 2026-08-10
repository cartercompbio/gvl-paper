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
    """Setup phase: one interval search per contig, folded down to exactly the
    pairs' (region, sample) cells.

    Structurally identical to `_svar_search` (SVAR v1): one cross-product
    search call per contig over (this contig's regions) x (its unique
    samples), then a per-pair diagonal extraction picking out the cells the
    pairs actually name. Both steps run inside the timed setup, exactly as
    SVAR v1's do.

    The fold is `HapRangesRect.select`, which produces genoray's FLAT
    `HapRanges` contract: one row per pair, no sample axis. The alternative --
    `_find_ranges`' `RangesBundle` -- cannot express that, since its
    `sample_cols` axis makes it a region x sample rectangle; covering a pair
    set with it costs either the whole cross-product in one gather or one
    gather per unique sample, and every such gather rebuilds the contig-wide
    dense union (`gather.rs::gather_ranges` -> `reader.dense_union()`).
    """
    by_contig: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for (c, s, e), smp in pairs:
        by_contig[c].append((s, e, smp))

    out: list[tuple[str, dict]] = []
    for contig, rows in by_contig.items():
        starts_list, ends_list, samples_list = zip(*rows)
        unique_samples = sorted(set(samples_list))
        sample_slot = {s: j for j, s in enumerate(unique_samples)}

        rect = sv._find_haps_ranges(
            contig,
            np.asarray(starts_list, dtype=np.int64),
            np.asarray(ends_list, dtype=np.int64),
            samples=np.asarray(unique_samples),
        )
        # Region r of the rectangle IS pair r -- every pair contributed its own
        # region row -- so the diagonal is (arange(n_pairs), that pair's slot).
        out.append((
            contig,
            rect.select(
                np.arange(len(rows)),
                np.fromiter(
                    (sample_slot[s] for s in samples_list),
                    dtype=np.intp,
                    count=len(rows),
                ),
            ),
        ))
    return out


def _svar2_gather(sv: SparseVar2, hap_ranges_by_contig: list[tuple[str, dict]]) -> None:
    """Read phase: tree-free read-bound replay of exactly the pair cells, one
    call per (batch, contig). The timed operation for throughput."""
    for contig, hr in hap_ranges_by_contig:
        sv._gather_haps_readbound(contig, hr)


def _svar2_n_calls(sv: SparseVar2, hap_ranges_by_contig: list[tuple[str, dict]]) -> int:
    """Variant-record count for one batch, computed OUTSIDE the timed regions.

    One extra (untimed) read-bound gather per contig, counting exactly what it
    emits for the pairs' cells: the var_key records plus the set dense presence
    bits. This is the SVAR2 analog of `_svar_search`'s
    `(flat_ends - flat_starts).sum()` -- the number of variant records the
    timed read copies -- and is the denominator every throughput number is
    divided by, so it must count the SAME cells the gather covers. Deriving it
    from the gather makes that identity structural rather than asserted.

    The obvious alternative, `sv.region_counts`, counts over the store's FULL
    cohort on the sample axis and is then indexed down to the named cells. It
    agrees exactly (guarded in test_bench_svar2.py) but is unusable at campaign
    scale: measured on 1kGP (3202 samples) at query_length=2048, one contig's
    688 regions cost 425 s, against 3.2 ms for the gather below -- and the
    q=2048 arm alone has 8192 regions per batch across 64 batches per
    replicate.
    """
    total = 0
    for contig, hr in hap_ranges_by_contig:
        d = sv._gather_haps_readbound(contig, hr)
        total += len(np.asarray(d["vk_pos"]))
        for cls in ("snp", "indel"):
            n_bits = int(np.asarray(d[f"dense_{cls}_present_off"])[-1])
            if n_bits:
                bits = np.asarray(d[f"dense_{cls}_present"], np.uint8)
                total += int(np.unpackbits(bits, bitorder="little")[:n_bits].sum())
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

        # AOT interval search + per-pair diagonal fold (cached ahead of
        # training); timed once -> setup_ns. n_calls is intentionally NOT
        # computed here -- see _svar2_n_calls.
        t0 = perf_counter_ns()
        ranges_per_batch = [_svar2_search(sv, pairs) for pairs in batches]
        setup_ns = perf_counter_ns() - t0

        # Variant-record count, computed OUTSIDE the timed setup region.
        payloads = [
            (hap_ranges, _svar2_n_calls(sv, hap_ranges))
            for hap_ranges in ranges_per_batch
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
