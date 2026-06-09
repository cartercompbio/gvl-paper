"""Steady-state sustained-loop timing for the variant-throughput benchmark.

The training-dataloader use case streams batches back-to-back for an entire
epoch, so the numba thread pool stays hot. Single-shot timing of a sub-ms gather
instead sees a bimodal distribution (true cost vs. true cost + a ~10 ms
thread-pool-wakeup penalty). This driver warms up, then times a tight loop that
cycles pre-prepared batches with nothing but list indexing between gathers, so
the pool never parks.

`gather(batch) -> int` executes one batch's gather and returns its call count.
Timing uses an injectable `now_ns` clock so the loop logic is unit-testable.
"""

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class StreamResult:
    distinct_calls: int  # calls over the distinct stream (one pass, no repeats)
    elapsed_ns: int  # single-pass-equivalent gather time at the steady-state rate
    n_measured: int  # number of timed gather iterations (includes cycle repeats)
    duration_ns: int  # actual wall time of the timed loop

    def n_calls_per_sec(self) -> float:
        if self.elapsed_ns == 0:
            return 0.0
        return self.distinct_calls / (self.elapsed_ns * 1e-9)


def prime(gather: Callable[[T], int], batches: Sequence[T], passes: int = 2) -> None:
    """Untimed warmup: cycle the batches `passes` times to prime threads/JIT/cache."""
    n = len(batches)
    for i in range(passes * n):
        gather(batches[i % n])


def drive_loop(
    gather: Callable[[T], int],
    batches: Sequence[T],
    *,
    min_seconds: float,
    min_batches: int,
    now_ns: Callable[[], int] = perf_counter_ns,
) -> tuple[int, int, int]:
    """Time a tight loop cycling `batches` until both bounds are met.

    Returns (total_calls, duration_ns, n_iterations). Stops once
    n_iterations >= min_batches AND elapsed >= min_seconds.
    """
    n = len(batches)
    min_ns = round(min_seconds * 1e9)
    total = 0
    i = 0
    elapsed = 0
    t0 = now_ns()
    while True:
        total += gather(batches[i % n])
        i += 1
        elapsed = now_ns() - t0
        if i >= min_batches and elapsed >= min_ns:
            break
    return total, elapsed, i


def run_stream(
    gather: Callable[[T], int],
    batches: Sequence[T],
    *,
    warmup: int = 2,
    min_seconds: float = 5.0,
    min_batches: int = 10,
    now_ns: Callable[[], int] = perf_counter_ns,
) -> StreamResult | None:
    """Warm up, then time a sustained gather loop; normalize to a single pass.

    `distinct_calls` is summed over one pass of the distinct batches (and that
    pass doubles as the first priming pass). `elapsed_ns` is the single-pass
    equivalent at the measured steady-state rate, so `distinct_calls/elapsed_ns`
    equals the rate even though the timed loop cycles (repeats) batches.
    """
    if not batches:
        return None

    distinct_calls = sum(gather(b) for b in batches)
    prime(gather, batches, passes=warmup)
    total, duration_ns, iters = drive_loop(
        gather, batches, min_seconds=min_seconds, min_batches=min_batches, now_ns=now_ns
    )
    rate = total / (duration_ns * 1e-9) if duration_ns > 0 else 0.0
    elapsed_ns = round(distinct_calls / rate * 1e9) if rate > 0 else 0
    return StreamResult(
        distinct_calls=distinct_calls,
        elapsed_ns=elapsed_ns,
        n_measured=iters,
        duration_ns=duration_ns,
    )
