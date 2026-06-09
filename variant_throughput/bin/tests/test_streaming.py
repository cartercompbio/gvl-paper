import pytest

from _streaming import StreamResult, drive_loop, prime, run_stream


def _clock(step: int = 1):
    """Deterministic ns clock advancing `step` ns per call."""
    t = {"n": 0}

    def now() -> int:
        t["n"] += step
        return t["n"]

    return now


def _counting_gather(calls_seen):
    """gather(batch) -> n_calls; batch is its own int call-count. Records calls."""

    def gather(batch: int) -> int:
        calls_seen.append(batch)
        return batch

    return gather


def test_prime_runs_passes_times_n_gathers():
    seen = []
    prime(_counting_gather(seen), [5, 7, 9], passes=2)
    # 2 passes over 3 batches = 6 gather calls, cycling in order
    assert seen == [5, 7, 9, 5, 7, 9]


def test_run_stream_empty_returns_none():
    assert run_stream(_counting_gather([]), [], min_seconds=0.0, min_batches=1) is None


def test_run_stream_distinct_calls_is_one_pass_sum():
    res = run_stream(
        _counting_gather([]),
        [10, 20, 30],
        warmup=0,
        min_seconds=0.0,
        min_batches=1,
        now_ns=_clock(),
    )
    assert isinstance(res, StreamResult)
    assert res.distinct_calls == 60  # 10+20+30, one pass, no repeats


def test_run_stream_min_batches_dominates():
    # min_seconds=0 so only min_batches bounds the loop.
    res = run_stream(
        _counting_gather([]),
        [10, 20],
        warmup=0,
        min_seconds=0.0,
        min_batches=5,
        now_ns=_clock(),
    )
    assert res.n_measured == 5  # stopped at min_batches


def test_run_stream_elapsed_ns_normalizes_under_cycle_repeats():
    # Coarse 1e9 ns/call clock so elapsed_ns does not round to 0. Batches [10,20],
    # min_batches=4 -> the timed loop cycles twice (10,20,10,20), total=60 over a
    # distinct pass of 30. elapsed_ns must be the single-pass equivalent: dur scaled
    # by distinct/total, so n_calls/elapsed_ns recovers the steady-state rate.
    res = run_stream(
        _counting_gather([]),
        [10, 20],
        warmup=0,
        min_seconds=0.0,
        min_batches=4,
        now_ns=_clock(step=10**9),
    )
    assert res.distinct_calls == 30
    assert res.n_measured == 4
    # duration = 4 timed iters * 1e9 ns = 4e9; rate = 60 / 4 s = 15 calls/s;
    # elapsed_ns = round(30 / 15 * 1e9) = 2e9.
    assert res.duration_ns == 4 * 10**9
    assert res.elapsed_ns == 2 * 10**9
    assert res.n_calls_per_sec() == pytest.approx(15.0)


def test_drive_loop_min_seconds_dominates():
    # 1 ns/call clock; min_batches=1, min_seconds tiny so the ns threshold ends it.
    # min_ns = round(min_seconds*1e9). Pick min_seconds=3e-9 -> min_ns=3.
    total, dur, iters = drive_loop(
        _counting_gather([]),
        [4],
        min_seconds=3e-9,
        min_batches=1,
        now_ns=_clock(),
    )
    # t0=1; iter1 elapsed=2-1=1 (<3); iter2 elapsed=3-1=2 (<3); iter3 elapsed=4-1=3 (>=3) -> stop
    assert iters == 3
    assert total == 12  # 3 iterations * 4 calls
    assert dur == 3
