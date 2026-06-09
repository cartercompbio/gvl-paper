# Variant-throughput steady-state streaming bench — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-shot per-replicate gather timing (which exposes a bimodal ~10 ms numba thread-pool-wakeup artifact in SVAR) with a hot-pool sustained-loop measurement that mirrors the GenVarLoader training dataloader.

**Architecture:** A shared `_streaming.py` helper drives a warmup-then-sustained-loop over a stream of pre-prepared batches, with an injectable clock for deterministic tests. `generate_pairs.py` emits a fixed-length batched pair stream (identical across formats). Each `bench_*.py` runs its ahead-of-time (AOT) step once (→ `setup_ns`) then replays only the gather in the timed loop (→ `elapsed_ns`). Throughput is normalized to a single pass over the distinct stream so one `n_calls` column stays consistent across both output panels. CSV schema and plotting code are unchanged.

**Tech Stack:** Python 3.12, polars, numpy, awkward, numba, genoray (`SparseVar`/`VCF`/`PGEN`), cyvcf2, cyclopts, pytest; orchestrated by typed Nextflow (`variant_throughput.nf`); run under pixi (default env).

**Spec:** `docs/superpowers/specs/2026-06-09-variant-throughput-steady-state-design.md`

**Conventions for every command below:** run from repo root `/carter/users/dlaub/projects/gvl-paper`; prefix Python/pytest with `pixi run` (default env). Nextflow `bin/` is on PATH at runtime, so modules import by bare name; tests replicate this via a `conftest.py` that adds `bin/` to `sys.path`.

---

## File structure

- Create `variant_throughput/bin/_streaming.py` — sustained-loop driver (`prime`, `drive_loop`, `run_stream`, `StreamResult`). Pure, clock-injectable, no I/O.
- Create `variant_throughput/bin/_pairs.py` — `compute_batch_size`, `split_pair_batches`. Pure helpers shared by `generate_pairs.py` and all four bench scripts.
- Create `variant_throughput/bin/tests/conftest.py` — puts `bin/` on `sys.path`.
- Create `variant_throughput/bin/tests/test_streaming.py`, `test_pairs.py`.
- Modify `variant_throughput/bin/generate_pairs.py` — emit a fixed batched stream with a `batch_id` column.
- Modify `variant_throughput/bin/bench_svar.py`, `bench_bcf.py`, `bench_pgen.py`, `bench_presubset_bcf.py` — AOT-once + sustained gather replay, both `throughput` and `memory` modes.
- Modify `variant_throughput/variant_throughput.nf` — new params (`min_seconds`, `min_batches`, `stream_batches`), drop `max_pairs`, thread params into `GENERATE_PAIRS`, `GENERATE_PAIRS_N`, and the eight `BENCH_*` processes.
- Modify `variant_throughput/configs/smoke.config` — small streaming params for the end-to-end smoke run.

---

## Task 1: `_streaming.py` sustained-loop driver

**Files:**
- Create: `variant_throughput/bin/_streaming.py`
- Create: `variant_throughput/bin/tests/conftest.py`
- Test: `variant_throughput/bin/tests/test_streaming.py`

- [ ] **Step 1: Create the test conftest**

Create `variant_throughput/bin/tests/conftest.py`:

```python
"""Put the parent `bin/` dir on sys.path so tests import sibling modules by bare
name (`_streaming`, `_pairs`) exactly as Nextflow does on PATH."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 2: Write the failing tests**

Create `variant_throughput/bin/tests/test_streaming.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pixi run pytest variant_throughput/bin/tests/test_streaming.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_streaming'`.

- [ ] **Step 4: Implement `_streaming.py`**

Create `variant_throughput/bin/_streaming.py`:

```python
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
    elapsed_ns = round(distinct_calls / rate) if rate > 0 else 0
    return StreamResult(
        distinct_calls=distinct_calls,
        elapsed_ns=elapsed_ns,
        n_measured=iters,
        duration_ns=duration_ns,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest variant_throughput/bin/tests/test_streaming.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add variant_throughput/bin/_streaming.py variant_throughput/bin/tests/conftest.py variant_throughput/bin/tests/test_streaming.py
git commit -m "feat(variant-throughput): sustained-loop streaming driver"
```

---

## Task 2: `_pairs.py` batch helpers

**Files:**
- Create: `variant_throughput/bin/_pairs.py`
- Test: `variant_throughput/bin/tests/test_pairs.py`

- [ ] **Step 1: Write the failing tests**

Create `variant_throughput/bin/tests/test_pairs.py`:

```python
import polars as pl

from _pairs import compute_batch_size, split_pair_batches


def test_compute_batch_size_inverse_scaling():
    assert compute_batch_size(2048, 2**24) == 8192
    assert compute_batch_size(2**24, 2**24) == 1
    # never below 1, even when query_length exceeds the budget
    assert compute_batch_size(2**25, 2**24) == 1


def test_split_pair_batches_groups_by_batch_id_in_order():
    df = pl.DataFrame(
        {
            "batch_id": [0, 0, 1],
            "contig": ["1", "1", "2"],
            "start": [10, 20, 30],
            "end": [12, 22, 32],
            "sample": ["s0", "s1", "s2"],
        }
    )
    batches = split_pair_batches(df)
    assert batches == [
        [(("1", 10, 12), "s0"), (("1", 20, 22), "s1")],
        [(("2", 30, 32), "s2")],
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest variant_throughput/bin/tests/test_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_pairs'`.

- [ ] **Step 3: Implement `_pairs.py`**

Create `variant_throughput/bin/_pairs.py`:

```python
"""Pair-stream helpers shared by generate_pairs.py and the bench_*.py scripts."""

import polars as pl

Pair = tuple[tuple[str, int, int], str]


def compute_batch_size(query_length: int, bp_budget: int) -> int:
    """Pairs per batch holding base-pairs-per-batch approximately constant.

    Mirrors the convention used by the memory-growth figure: a batch covers at
    most `bp_budget` base pairs, so batch size scales inversely with query length.
    """
    return max(1, bp_budget // query_length)


def split_pair_batches(group: pl.DataFrame) -> list[list[Pair]]:
    """Split one replicate's rows into batches by `batch_id` (ascending).

    Each pair is ((contig, start, end), sample). Row order within a batch is
    preserved.
    """
    batches: list[list[Pair]] = []
    for _, batch in group.sort("batch_id").group_by("batch_id", maintain_order=True):
        batches.append(
            [
                ((row["contig"], int(row["start"]), int(row["end"])), row["sample"])
                for row in batch.iter_rows(named=True)
            ]
        )
    return batches
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest variant_throughput/bin/tests/test_pairs.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add variant_throughput/bin/_pairs.py variant_throughput/bin/tests/test_pairs.py
git commit -m "feat(variant-throughput): batch-size + batch-split helpers"
```

---

## Task 3: `generate_pairs.py` emits a batched stream

**Files:**
- Modify: `variant_throughput/bin/generate_pairs.py`

Replace the random-count, total-length-capped emission with a fixed stream of
`stream_batches * batch_size` pairs per replicate, tagged with `batch_id`.

- [ ] **Step 1: Replace the `bench()` signature params**

In `variant_throughput/bin/generate_pairs.py`, change the signature:

```python
def bench(
    svar: Path,
    fai: Path,
    query_length: int,
    output: Path,
    seed: int = 0,
    n_replicates: int = 5,
    stream_batches: int = 64,
    bp_budget: int = 2**24,
):
```

(Removed `max_pairs` and `max_total_length`; added `stream_batches` and `bp_budget`.)

- [ ] **Step 2: Import the batch-size helper**

At the top of `generate_pairs.py`, below `from cyclopts import run`, add:

```python
from _pairs import compute_batch_size
```

- [ ] **Step 3: Replace the per-replicate sampling loop**

Replace the emission loop (the `for rep in range(n_replicates):` block that uses
`target = rng.randint(1, max_pairs)` and `max_total_length`) and the column
collection so it emits a fixed batched stream. The new block:

```python
    batch_size = compute_batch_size(query_length, bp_budget)
    n_pairs = stream_batches * batch_size

    replicates: list[int] = []
    batch_ids: list[int] = []
    contigs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    samples: list[str] = []

    for rep in range(n_replicates):
        for k in range(n_pairs):
            region = sample_region(allowed, query_length, rng)
            if region is None:
                break
            contig, start, end = region
            sample = rng.choice(available_samples)
            replicates.append(rep)
            batch_ids.append(k // batch_size)
            contigs.append(contig)
            starts.append(start)
            ends.append(end)
            samples.append(sample)

    pl.DataFrame(
        {
            "replicate": pl.Series(replicates, dtype=pl.UInt16),
            "batch_id": pl.Series(batch_ids, dtype=pl.UInt32),
            "contig": contigs,
            "start": pl.Series(starts, dtype=pl.Int64),
            "end": pl.Series(ends, dtype=pl.Int64),
            "sample": samples,
        }
    ).write_parquet(output)
```

Leave `load_gap_intervals`, `build_allowed_intervals`, and `sample_region`
unchanged.

- [ ] **Step 4: Verify it parses and the CLI loads**

Run: `pixi run python variant_throughput/bin/generate_pairs.py --help`
Expected: cyclopts help text listing `--stream-batches` and `--bp-budget`, with no `--max-pairs`.

- [ ] **Step 5: Commit**

```bash
git add variant_throughput/bin/generate_pairs.py
git commit -m "feat(variant-throughput): emit fixed batched pair stream with batch_id"
```

---

## Task 4: `bench_svar.py` — AOT search + streamed pack

**Files:**
- Modify: `variant_throughput/bin/bench_svar.py`

SVAR caches the variant-index search ahead of time, so the timed loop replays
only `_svar_pack`.

- [ ] **Step 1: Update imports and `bench()` signature**

In `variant_throughput/bin/bench_svar.py`, add helper imports near the top
(after `from numpy.typing import NDArray`):

```python
from _pairs import split_pair_batches
from _streaming import drive_loop, prime, run_stream
```

Change the `bench()` signature to add streaming knobs:

```python
def bench(
    pairs_parquet: Path,
    svar: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    use_custom_pack: bool = True,
    n_samples: int = 0,
    min_seconds: float = 5.0,
    min_batches: int = 10,
):
```

- [ ] **Step 2: Replace the per-replicate body**

Replace the entire `for rep_val, group in df.group_by(...)` loop body (both the
`throughput` and `memory` branches) with:

```python
    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        batches = split_pair_batches(group)
        if not batches:
            continue

        # AOT index search (cached ahead of training); timed once -> setup_ns.
        t0 = perf_counter_ns()
        cached = [_svar_search(_svar, pairs) for pairs in batches]
        setup_ns = perf_counter_ns() - t0
        # payload per batch: (flat_starts, flat_ends, n_calls)
        payloads = [(fs, fe, nc) for (fs, fe, nc) in cached]
        n_pairs = sum(len(pairs) for pairs in batches)

        def gather(p) -> int:
            _svar_pack(_svar, p[0], p[1], use_custom_pack)
            return p[2]

        if mode == "throughput":
            res = run_stream(
                gather, payloads, min_seconds=min_seconds, min_batches=min_batches
            )
            rows_out.append({
                "dataset": dataset or svar.name,
                "method": "svar",
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

            distinct_calls = sum(p[2] for p in payloads)
            prime(gather, payloads)
            with PeakRssSampler() as s:
                drive_loop(
                    gather, payloads, min_seconds=min_seconds, min_batches=min_batches
                )
            rows_out.append({
                "dataset": dataset or svar.name,
                "method": "svar",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": n_pairs,
                "n_calls": distinct_calls,
                "peak_rss_bytes": s.peak,
            })
```

Keep the numba JIT warmup block, `_svar_search`, `_svar_pack`, `to_packed_custom`,
`_gather_parallel`, the `df = pl.read_parquet(...)`/`q_len = ...` lines, and the
final `pl.DataFrame(rows_out).write_csv(output)` unchanged.

- [ ] **Step 3: Verify it parses and the CLI loads**

Run: `pixi run python variant_throughput/bin/bench_svar.py --help`
Expected: help text including `--min-seconds` and `--min-batches`; no error.

- [ ] **Step 4: Commit**

```bash
git add variant_throughput/bin/bench_svar.py
git commit -m "feat(variant-throughput): SVAR AOT search + streamed pack replay"
```

---

## Task 5: `bench_bcf.py` — streamed read

**Files:**
- Modify: `variant_throughput/bin/bench_bcf.py`

BCF has no separable AOT cache; the timed loop replays the full locate+read.

- [ ] **Step 1: Update imports and signature**

In `variant_throughput/bin/bench_bcf.py`, add after `from cyclopts import run`:

```python
from _pairs import split_pair_batches
from _streaming import drive_loop, prime, run_stream
```

Change the signature to:

```python
def bench(
    pairs_parquet: Path,
    bcf: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
    min_seconds: float = 5.0,
    min_batches: int = 10,
):
```

- [ ] **Step 2: Replace the per-replicate body**

Replace the entire `for rep_val, group in df.group_by(...)` loop body with:

```python
    _bcf = VCF(bcf, with_gvi_index=False)

    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        batches = split_pair_batches(group)
        if not batches:
            continue
        n_pairs = sum(len(pairs) for pairs in batches)

        def gather(pairs) -> int:
            nonlocal _bcf
            n = 0
            for (contig, start, end), sample in pairs:
                _bcf = _bcf.set_samples(sample)
                genos = _bcf.read(contig, start, end, mode=_bcf.Genos8)
                n += int((genos > 0).sum())
            return n

        if mode == "throughput":
            res = run_stream(
                gather, batches, min_seconds=min_seconds, min_batches=min_batches
            )
            rows_out.append({
                "dataset": dataset or bcf.name,
                "method": "bcf",
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
                "dataset": dataset or bcf.name,
                "method": "bcf",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": n_pairs,
                "n_calls": distinct_calls,
                "peak_rss_bytes": s.peak,
            })
```

Keep the `import numpy as np` / `import polars as pl` / `from genoray import VCF`
imports, the `df`/`q_len` lines, and the final `write_csv` unchanged.

- [ ] **Step 3: Verify it parses and the CLI loads**

Run: `pixi run python variant_throughput/bin/bench_bcf.py --help`
Expected: help text including `--min-seconds` and `--min-batches`; no error.

- [ ] **Step 4: Commit**

```bash
git add variant_throughput/bin/bench_bcf.py
git commit -m "feat(variant-throughput): BCF streamed read replay"
```

---

## Task 6: `bench_pgen.py` — streamed read

**Files:**
- Modify: `variant_throughput/bin/bench_pgen.py`

- [ ] **Step 1: Update imports and signature**

In `variant_throughput/bin/bench_pgen.py`, add after `from cyclopts import run`:

```python
from _pairs import split_pair_batches
from _streaming import drive_loop, prime, run_stream
```

Change the signature to:

```python
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
```

- [ ] **Step 2: Replace the per-replicate body**

Replace the entire `for rep_val, group in df.group_by(...)` loop body with:

```python
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
```

Keep the `import polars as pl` / `from genoray import PGEN` imports, the
`df`/`q_len` lines, and the final `write_csv` unchanged.

- [ ] **Step 3: Verify it parses and the CLI loads**

Run: `pixi run python variant_throughput/bin/bench_pgen.py --help`
Expected: help text including `--min-seconds` and `--min-batches`; no error.

- [ ] **Step 4: Commit**

```bash
git add variant_throughput/bin/bench_pgen.py
git commit -m "feat(variant-throughput): PGEN streamed read replay"
```

---

## Task 7: `bench_presubset_bcf.py` — AOT subset + streamed read

**Files:**
- Modify: `variant_throughput/bin/bench_presubset_bcf.py`

The `bcftools` subset is the AOT step (→ `setup_ns`); the timed loop replays the
`cyvcf2` read of the cached temp BCFs.

- [ ] **Step 1: Update imports and signature**

In `variant_throughput/bin/bench_presubset_bcf.py`, add after
`from cyclopts import run`:

```python
from _pairs import split_pair_batches
from _streaming import drive_loop, prime, run_stream
```

Change the `bench()` signature to:

```python
def bench(
    pairs_parquet: Path,
    bcf: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
    min_seconds: float = 5.0,
    min_batches: int = 10,
):
```

- [ ] **Step 2: Replace the per-replicate body**

Replace the entire `for rep_val, group in df.group_by(...)` loop body (the whole
`tmp_paths = []` / `try: ... finally:` block) with:

```python
    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        batches = split_pair_batches(group)
        if not batches:
            continue
        n_pairs = sum(len(pairs) for pairs in batches)

        # AOT: pre-subset each batch into temp BCFs (cached); timed once -> setup_ns.
        t0 = perf_counter_ns()
        batch_tmp = [_subset_pairs(bcf, pairs, tmp_dir) for pairs in batches]
        setup_ns = perf_counter_ns() - t0

        def gather(tmp_paths) -> int:
            return _read_subsets(tmp_paths)

        try:
            if mode == "throughput":
                res = run_stream(
                    gather, batch_tmp, min_seconds=min_seconds, min_batches=min_batches
                )
                rows_out.append({
                    "dataset": dataset or bcf.name,
                    "method": "presubset_bcf",
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

                distinct_calls = sum(gather(tp) for tp in batch_tmp)
                prime(gather, batch_tmp)
                with PeakRssSampler() as s:
                    drive_loop(
                        gather, batch_tmp, min_seconds=min_seconds, min_batches=min_batches
                    )
                rows_out.append({
                    "dataset": dataset or bcf.name,
                    "method": "presubset_bcf",
                    "query_length": q_len,
                    "n_samples": int(n_samples),
                    "replicate": int(rep),
                    "n_pairs": n_pairs,
                    "n_calls": distinct_calls,
                    "peak_rss_bytes": s.peak,
                })
        finally:
            for tmp_paths in batch_tmp:
                for path in tmp_paths:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
```

Keep `_subset_pairs`, `_read_subsets`, the `import polars as pl`, the
`df`/`q_len`/`tmp_dir` lines, and the final `write_csv` unchanged.

- [ ] **Step 3: Verify it parses and the CLI loads**

Run: `pixi run python variant_throughput/bin/bench_presubset_bcf.py --help`
Expected: help text including `--min-seconds` and `--min-batches`; no error.

- [ ] **Step 4: Commit**

```bash
git add variant_throughput/bin/bench_presubset_bcf.py
git commit -m "feat(variant-throughput): PRESUB-BCF AOT subset + streamed read replay"
```

---

## Task 8: Wire new params through `variant_throughput.nf` and smoke config

**Files:**
- Modify: `variant_throughput/variant_throughput.nf`
- Modify: `variant_throughput/configs/smoke.config`

- [ ] **Step 1: Update the `params` block**

In `variant_throughput/variant_throughput.nf`, in the top-level `params { ... }`
block, remove the `max_pairs` line and add the streaming params. The relevant
lines become:

```groovy
    seed: Integer = 0
    n_replicates: Integer = 5
    stream_batches: Integer = 64
    bp_budget: Integer = 16777216
    min_seconds: Double = 5.0
    min_batches: Integer = 10
    query_lengths: List<Integer> = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216]
    use_custom_pack: Boolean = true
    sample_sizes: List<Integer> = [10, 32, 100, 316, 1000, 3202]
    n_sweep_query_length: Integer = 131072
    sample_seed: Integer = 0
```

(`max_total_length` is replaced by `bp_budget`; `max_pairs` is removed.)

- [ ] **Step 2: Update the `GENERATE_PAIRS` process inputs and script**

In `process GENERATE_PAIRS`, replace the `max_pairs`/`max_total_length` input
declarations and script flags. The `input:` block becomes:

```groovy
    input:
    query_length: Integer
    n_replicates: Integer
    stream_batches: Integer
    seed: Integer
    bp_budget: Integer
    n_samples_full: Integer
```

and the `script:` here-doc becomes:

```groovy
    """
    generate_pairs.py \\
      ${params.svar} \\
      ${params.fai} \\
      ${query_length} \\
      pairs_${query_length}.parquet \\
      --seed ${seed} \\
      --n-replicates ${n_replicates} \\
      --stream-batches ${stream_batches} \\
      --bp-budget ${bp_budget}
    """
```

- [ ] **Step 3: Update the `GENERATE_PAIRS` call in the workflow**

In `workflow { ... }`, update the `GENERATE_PAIRS(...)` invocation arguments to
match the new input order:

```groovy
    pairs_raw = GENERATE_PAIRS(
        lengths,
        params.n_replicates,
        params.stream_batches,
        params.seed,
        params.bp_budget,
        n_full,
    )
```

- [ ] **Step 4: Update `GENERATE_PAIRS_N` process inputs and script**

In `process GENERATE_PAIRS_N`, replace the `max_pairs`/`max_total_length` input
declarations. The `input:` block becomes:

```groovy
    input:
    t: SubsetTriple
    query_length: Integer
    n_replicates: Integer
    stream_batches: Integer
    seed: Integer
    bp_budget: Integer
```

and the `script:` here-doc becomes:

```groovy
    """
    generate_pairs.py \\
      ${t.svar} \\
      ${params.fai} \\
      ${query_length} \\
      pairs_N${t.n}.parquet \\
      --seed ${seed} \\
      --n-replicates ${n_replicates} \\
      --stream-batches ${stream_batches} \\
      --bp-budget ${bp_budget}
    """
```

- [ ] **Step 5: Update the `GENERATE_PAIRS_N` call in the workflow**

```groovy
    n_pairs = GENERATE_PAIRS_N(
        triples,
        params.n_sweep_query_length,
        params.n_replicates,
        params.stream_batches,
        params.seed,
        params.bp_budget,
    )
```

- [ ] **Step 6: Append streaming flags to all eight `BENCH_*` script blocks**

For each of `BENCH_SVAR_THROUGHPUT`, `BENCH_SVAR_MEMORY`, `BENCH_BCF_THROUGHPUT`,
`BENCH_BCF_MEMORY`, `BENCH_PGEN_THROUGHPUT`, `BENCH_PGEN_MEMORY`,
`BENCH_PRESUBSET_BCF_THROUGHPUT`, `BENCH_PRESUBSET_BCF_MEMORY`, append the two
flags to the script invocation (the last line before the closing `"""`). For
example, in `BENCH_SVAR_THROUGHPUT` the script becomes:

```groovy
    """
    bench_svar.py \\
      ${p.pairs} \\
      ${p.svar} \\
      svar_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      ${pack_flag} \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
    """
```

Apply the same two trailing flags (`--min-seconds ${params.min_seconds}
--min-batches ${params.min_batches}`) to the other seven `BENCH_*` script blocks,
preserving each block's existing flags (`--mode memory` where applicable, and
`${pack_flag}` only in the two SVAR processes).

- [ ] **Step 7: Update the smoke config**

Replace the contents of `variant_throughput/configs/smoke.config` with:

```groovy
params {
    query_lengths = [2048, 16384]
    n_replicates = 2
    stream_batches = 4
    sample_sizes = [2, 5]
    n_sweep_query_length = 8192
    bp_budget = 65536
    min_seconds = 0.5
    min_batches = 3
}
```

- [ ] **Step 8: Verify the Nextflow script parses**

Run: `cd variant_throughput && pixi run nextflow run variant_throughput.nf -c configs/smoke.config -stub-run; cd ..`
Expected: the DSL2 graph compiles and stub-executes without a syntax/param error (stub-run does not invoke the scripts). If `-stub-run` reports missing `stub:` blocks, instead run a config-parse check: `cd variant_throughput && pixi run nextflow config -c configs/smoke.config >/dev/null && echo PARSE_OK; cd ..` — expected `PARSE_OK`.

- [ ] **Step 9: Commit**

```bash
git add variant_throughput/variant_throughput.nf variant_throughput/configs/smoke.config
git commit -m "feat(variant-throughput): thread streaming params through pipeline + smoke config"
```

---

## Task 9: End-to-end smoke run and artifact verification

**Files:** none modified — this task validates the integrated pipeline against real (small) data.

**Precondition:** a real SVAR/BCF/PGEN dataset is reachable (the paths in
`configs/1kgp.config`, or any small local dataset). The smoke config restricts to
two short query lengths and tiny cohorts, so it is fast.

- [ ] **Step 1: Run the smoke pipeline**

Run:
```bash
cd variant_throughput && pixi run nextflow run variant_throughput.nf -c configs/1kgp.config -c configs/smoke.config -resume; cd ..
```
Expected: all `GENERATE_PAIRS*`, `BENCH_*`, `COMBINE_*`, and `PLOT_*` processes
complete; `results/` gains `svar_throughput.csv`, `plot.png`, `n_plot.png`,
`setup_plot.png`, and the memory CSVs/plots.

- [ ] **Step 2: Verify the throughput CSV schema is unchanged**

Run:
```bash
pixi run python -c "import polars as pl; df=pl.read_csv('variant_throughput/results/svar_throughput.csv'); print(df.columns); print(df.head())"
```
Expected columns exactly: `dataset, method, query_length, n_samples, replicate, n_pairs, n_calls, elapsed_ns, setup_ns`.

- [ ] **Step 3: Verify the SVAR bimodal artifact is gone**

Run:
```bash
pixi run python -c "
import polars as pl
df = pl.read_csv('variant_throughput/results/svar_throughput.csv')
# Per (query_length, n_samples) cell, the spread of elapsed_ns across replicates
# should be tight (no ~10ms-vs-~0.2ms split). Report max/min ratio per cell.
g = (df.group_by('query_length','n_samples')
       .agg((pl.col('elapsed_ns').max()/pl.col('elapsed_ns').min()).alias('spread')))
print(g.sort('spread', descending=True).head())
assert g['spread'].max() < 10, 'elapsed_ns still bimodal within a cell'
print('OK: SVAR elapsed_ns unimodal within cells')
"
```
Expected: `OK: SVAR elapsed_ns unimodal within cells` (the prior artifact produced ~50–100× within-cell spreads; steady-state replicate variance should be well under 10×).

- [ ] **Step 4: Sanity-check `n_calls/elapsed_ns == n_calls/setup_ns` consistency**

Run:
```bash
pixi run python -c "
import polars as pl
df = pl.read_csv('variant_throughput/results/svar_throughput.csv')
assert (df['n_calls'] > 0).all()
assert (df['elapsed_ns'] > 0).all()
assert (df['setup_ns'] > 0).all()
print('OK: positive n_calls/elapsed_ns/setup_ns for all SVAR rows')
"
```
Expected: `OK: positive n_calls/elapsed_ns/setup_ns for all SVAR rows`.

- [ ] **Step 5: Run the full unit-test suite**

Run: `pixi run pytest variant_throughput/bin/tests/ -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit any results regenerated during the smoke run (optional)**

The smoke run overwrites `results/*` with tiny smoke data. Do NOT commit smoke
results as the manuscript figures. If `results/` is tracked and dirty, restore it:
```bash
git checkout -- variant_throughput/results
```
The real regeneration (full `configs/1kgp.config` run without the smoke overlay)
is a separate operational step, not part of this plan.

---

## Self-review notes

- **Spec coverage:** stream+batch_id (Task 3), bp-budget batch size (Task 2/3),
  identical stream across formats (same parquet, Tasks 4–7), AOT/replay split per
  format (Tasks 4–7), warmup + sustained loop with `min_seconds=5`/`min_batches=10`
  (Task 1, wired Task 8), single-pass `n_calls` normalization (Task 1
  `run_stream`), memory mode streaming (Tasks 4–7), unchanged CSV schema + plotting
  (verified Task 9 Step 2), new nf params + dropped `max_pairs` (Task 8). All spec
  sections map to a task.
- **Out-of-scope** items (cold-cache regime, batch-size sweep, plotting changes,
  committing regenerated manuscript results) are explicitly excluded in Task 9
  Step 6 and untouched elsewhere.
- **Type/name consistency:** `run_stream`, `drive_loop`, `prime`, `StreamResult`
  (with `.distinct_calls`, `.elapsed_ns`, `.n_measured`, `.duration_ns`,
  `.n_calls_per_sec()`); `compute_batch_size`, `split_pair_batches`; param names
  `min_seconds`, `min_batches`, `stream_batches`, `bp_budget` are used identically
  across `_streaming.py`, `_pairs.py`, the four bench scripts, and the nf wiring.
```
