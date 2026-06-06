"""Shared byte-accounting + timed measurement loop for the throughput benchmarks.

Imported by benchmark_haps.py and benchmark_tracks.py. No genvarloader/torch
imports here so the math + control flow stay unit-testable under any env.

The measurement loop matches the bin_gvl061 / bin_gvl027 throughput convention:
bytes are accumulated for every batch with n_yielded >= burn_in, and the timer
starts at the burn_in-th batch. This makes throughput (MiB/s) directly
comparable to results/{hap,track}_results.csv.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Iterable, Optional

THROUGHPUT_HEADER = (
    "dataset,backend,dl_mode,threads,seqlen,batch_size,"
    "n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)\n"
)
MEMORY_HEADER = (
    "dataset,backend,dl_mode,threads,seqlen,batch_size,"
    "avg_rss_bytes,peak_rss_bytes\n"
)


@dataclass
class CellResult:
    n_measured: int
    total_bytes: int
    duration_ns: int


def n_bytes(batch) -> int:
    """Total bytes in a batch, supporting numpy arrays and torch-like tensors."""
    if hasattr(batch, "numel"):  # torch.Tensor (numpy ndarrays have no .numel)
        return int(batch.numel()) * int(batch.element_size())
    return int(batch.size) * int(batch.itemsize)  # numpy ndarray


def mib_per_s(total_bytes: int, seconds: float) -> float:
    """Throughput in MiB/s. Matches bin_gvl061's convention."""
    return total_bytes / seconds / 2**20


def measure_cell(
    dl: Iterable,
    *,
    burn_in: int,
    n_batches: int,
    time_limit_ns: int,
    min_batches: int,
    now_ns: Callable[[], int] = perf_counter_ns,
) -> Optional[CellResult]:
    """Time one (threads, batch_size) cell over a re-iterable dataloader.

    Accumulates bytes for batches with n_yielded >= burn_in; the timer t_start is
    reset at the burn_in-th batch. Stops once either n_batches measured batches
    have been seen or (min_batches reached AND the wall-clock limit elapsed).

    Returns CellResult, or None if an epoch yields zero batches (empty-epoch
    guard — record the cell as NaN and never spin the `while not done` loop
    forever; root cause of the 0.26.0 hang).
    """
    n_yielded = 0
    n_measured = 0
    total_bytes = 0
    t_start = now_ns()
    done = False
    while not done:
        epoch_count = 0
        for batch in dl:
            epoch_count += 1
            if n_yielded == burn_in:
                t_start = now_ns()
            if n_yielded >= burn_in:
                n_measured += 1
                total_bytes += n_bytes(batch)
                elapsed_ns = now_ns() - t_start
                if n_yielded + 1 >= burn_in + n_batches or (
                    n_measured >= min_batches and elapsed_ns >= time_limit_ns
                ):
                    done = True
                    n_yielded += 1
                    break
            n_yielded += 1
        if epoch_count == 0:  # empty-epoch guard
            return None
    duration_ns = now_ns() - t_start
    return CellResult(n_measured=n_measured, total_bytes=total_bytes, duration_ns=duration_ns)
