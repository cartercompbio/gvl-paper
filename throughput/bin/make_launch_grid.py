#!/usr/bin/env python3
"""
Generate a launch grid JSON for throughput benchmarks.

Outputs a JSON array of {length, threads, batch_size, n_batches} objects
for consumption by Nextflow's channel.splitJson. The workflow joins these
with dataset outputs on `length` to produce full config tuples.
"""

from __future__ import annotations

import math
from pathlib import Path

import cyclopts


def make_grid(
    *,
    lengths: list[int],
    min_npb: int | None,
    max_npb: int,
    n_threads_powers: range = range(7),
) -> list[dict]:
    """Build the benchmark launch grid for given lengths and params."""
    rows: list[dict] = []
    for length in lengths:
        if length <= 0:
            continue
        min_val = min_npb if min_npb is not None else length
        min_npb_ = max(min_val, length)
        min_bs = max(1, round(math.log2(min_npb_ / length)))
        max_bs = round(math.log2(max_npb / length))
        n_threads = [2**p for p in n_threads_powers]
        for bs_exp in range(min_bs, max_bs + 1):
            bs = 2**bs_exp
            npb = length * bs
            npb_safe = max(1, npb)
            n_batches = max(10, ceil_idiv(2**29, npb_safe))
            for t in n_threads:
                rows.append(
                    {
                        "length": length,
                        "threads": t,
                        "batch_size": bs,
                        "n_batches": n_batches,
                    }
                )
    return rows


def ceil_idiv(a: int, b: int) -> int:
    return -(-a // b)


def main(
    length: int,
    min_npb: int | None = None,
    max_npb: int = 2**33,
    n_threads_powers: list[int] | None = None,
    output: Path | None = None,
    test: bool = False,
):
    import polars as pl

    """Generate launch grid CSV for throughput benchmarks.

    With ``--test``, emit a tiny grid (threads in {1, 8}, batch_size in
    {1, 32}, n_batches=10) for quick smoke runs.
    """
    if test:
        grid = [
            {"threads": t, "batch_size": bs, "n_batches": 10}
            for t in (1, 8)
            for bs in (1, 32)
        ]
        if output is None:
            output = Path.cwd() / f"grid_{length}.csv"
        pl.from_dicts(grid).write_csv(output)
        return

    if n_threads_powers is None:
        n_threads_powers = list(range(7))

    grid: list[dict] = []
    min_val = min_npb if min_npb is not None else length
    min_npb_ = max(min_val, length)
    min_bs = max(1, round(math.log2(min_npb_ / length)))
    max_bs = round(math.log2(max_npb / length))
    n_threads = [2**p for p in n_threads_powers]
    for bs_exp in range(min_bs, max_bs + 1):
        bs = 2**bs_exp
        npb = length * bs
        npb_safe = max(1, npb)
        n_batches = max(10, ceil_idiv(2**29, npb_safe))
        for t in n_threads:
            grid.append({"threads": t, "batch_size": bs, "n_batches": n_batches})

    if output is None:
        output = Path.cwd() / f"grid_{length}.csv"
    pl.from_dicts(grid).write_csv(output)


if __name__ == "__main__":
    cyclopts.run(main)
