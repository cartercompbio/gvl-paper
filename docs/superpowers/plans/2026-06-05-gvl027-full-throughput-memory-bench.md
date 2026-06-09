# GVL 0.27.0 full throughput + memory bench (SVAR-backed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full `(threads × batch_size)` manuscript throughput grid plus a peak/avg RSS memory pass for GenVarLoader 0.27.0 across three datasets (1kgp, tcga-atac, ukbb) for haplotype and track dataloading, all SVAR-backed with genoray 2.9.0 symbolic/breakend filtering, via the production Nextflow harness, and confirm throughput parity vs. the 0.6.1 baseline as an internal sanity check.

**Architecture:** The existing `benchmark.nf` pipeline (`BENCH_SVAR_CONVERT → BENCH_WRITE_DATASET → BENCH_HAPS / BENCH_TRACKS`) is extended with two new dimensions — SVAR-only writes and a `dl_mode ∈ {none, buffered}` cross — and the two benchmark scripts are rewritten to emit a bytes-accounted, MiB/s schema directly comparable to the 0.6.1 baseline CSVs. genoray 2.9.0's hap-safe filter (`~is_symbolic & ~is_breakend`) is applied at SVAR conversion. A fresh `gvl027` Nextflow profile activates the `bench027` pixi env on each SLURM job. Throughput and memory are two pipeline invocations toggled by `--measure_memory`.

**Tech Stack:** Nextflow 26.04 typed DSL2, pixi (`bench027` env: genvarloader 0.27.0, genoray 2.9.0, CPU torch, polars, pytest), Python 3.12, cyclopts CLIs, SLURM (carter-cn-04).

---

## Background facts the implementer must know

These are verified against the installed code on `2026-06-05`. Do **not** re-derive them.

- **`bin/` is on Nextflow's PATH automatically.** Scripts in `hap_track_throughput/bin/` import sibling modules by bare name (e.g. `from _mem_sampler import PeakRssSampler`). A new shared module placed in `bin/` is therefore importable the same way.
- **0.6.1 baseline CSVs** (`results/hap_results.csv`, `results/track_results.csv`) have columns exactly: `dataset,threads,seqlen,batch_size,throughput (MiB/s)`. `dataset` values are lowercase/hyphenated: `1kgp`, `tcga-atac`, `ukbb`. `hap_results.csv` covers all three datasets; `track_results.csv` covers only `tcga-atac`. Seqlens present: `2048, 16384, 131072, 1048576`.
- **The nf configs set `params.dataset` to UPPERCASE/underscore** (`1KGP`, `TCGA_ATAC`, `UKBB`). The benchmark scripts write that value verbatim into the `dataset` CSV column. The parity join therefore needs case/`_`→`-` normalization (handled in Task 9).
- **The current `bin/benchmark_haps.py` / `benchmark_tracks.py`** emit `...,n_batches_measured,duration` with **no byte count and no MiB/s** — not directly comparable to the baseline. Task 3/4 fix this.
- **genoray 2.9.0** (installed in `bench027`) exposes `genoray.exprs.is_symbolic` and `genoray.exprs.is_breakend` (public). The hap-safe filter `~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend` composes **without** `import polars`. Verified: it drops `<DEL>`, `G[chr2:321[`, `.TGCA` and keeps `A`, `AT`.
  - `genoray.PGEN(path, filter=<expr>)` — PGEN takes the polars expr only.
  - `genoray.VCF(path, filter=<callable>, pl_filter=<expr>)` — VCF requires **both** a `cyvcf2.Variant→bool` callable **and** the matching polars expr (both-or-neither invariant enforced in `VCF.__init__`).
  - The breakend ALT regex (for the VCF callable, to avoid drift) is `r"[\[\]]|^\.[A-Za-z]|[A-Za-z]\.$"`.
  - `SparseVar.from_vcf` / `from_pgen` inherit and apply the source reader's filter, so filtering the `VCF`/`PGEN` filters the SVAR.
- **`Dataset.to_dataloader(mode="buffered", buffer_bytes=...)`** in 0.27.0: buffered + haplotypes requires `with_settings(deterministic=True)`; construction raises `ValueError` when a single mini-batch exceeds `buffer_bytes`. Empty epochs (0 batches) must be guarded against (root cause of the 0.26.0 hang; 0.27.0 fixed the behavior but the guard stays defensive).
- **Default/`bench` env genoray is currently 2.3.3**; `bench027` resolves genoray 2.9.0 transitively. Task 1 pins 2.9.0 explicitly in both.
- **No `nextflow.config` exists** under `hap_track_throughput/` yet. Task 8 creates it.

## File structure (what changes)

| File | Responsibility | Action |
|---|---|---|
| `pixi.toml` | Pin `genoray ==2.9.0` in `bench027` + `bench` | Modify |
| `hap_track_throughput/bin/_bench_common.py` | Shared byte-accounting + throughput-cell measurement loop (DRY across haps/tracks); empty-epoch guard | Create |
| `hap_track_throughput/bin/tests/conftest.py` | Put `bin/` on `sys.path` for pytest | Create |
| `hap_track_throughput/bin/tests/test_bench_common.py` | Unit tests for `_bench_common` | Create |
| `hap_track_throughput/bin/benchmark_haps.py` | Haps throughput+memory, dl_mode, new schema | Rewrite |
| `hap_track_throughput/bin/benchmark_tracks.py` | Tracks throughput+memory, dl_mode, new schema | Rewrite |
| `hap_track_throughput/bin/benchmark_svar_convert.py` | Apply hap-safe genoray filter at SVAR conversion | Modify |
| `hap_track_throughput/bin/_genoray_filter.py` | Shared hap-safe filter builder (expr + VCF callable) | Create |
| `hap_track_throughput/bin/tests/test_genoray_filter.py` | Unit tests for the filter builder | Create |
| `hap_track_throughput/bench_svar_vcf_plink.py` | Correctness-only filter fix to `VCF()`/`PGEN()` | Modify |
| `hap_track_throughput/benchmark.nf` | SVAR-only, dl_mode dimension, filter params, results_dir | Modify |
| `hap_track_throughput/nextflow.config` | `gvl027` profile (env activation, slurm executor) | Create |
| `hap_track_throughput/bin_gvl027/compare_to_baseline.py` | Read nf layout, normalize dataset, throughput parity + buffered standalone + memory absolute | Rewrite |
| `hap_track_throughput/bin_gvl027/tests/test_compare_helpers.py` | Unit tests for compare path/name helpers | Create |
| `hap_track_throughput/run_full_bench.sbatch` | Drive throughput + memory nf invocations per dataset | Create |
| `CLAUDE.md` | Document the full-bench harness + `gvl027` profile | Modify |

---

## Task 1: Pin genoray 2.9.0 in pixi

**Files:**
- Modify: `pixi.toml` (`[feature.bench027.pypi-dependencies]`, `[feature.bench.pypi-dependencies]`)

genoray 2.9.0 is what `bench027` already resolves and where the `--no-symbolic`/`--no-breakend` filters live. Pin it explicitly so it can't drift, and pin it in `bench`/default too so `bench_svar_vcf_plink.py` imports a consistent API.

- [ ] **Step 1: Add explicit genoray pin to `bench027`**

In `pixi.toml`, under `[feature.bench027.pypi-dependencies]`, add a line after `genvarloader = "==0.27.0"` (and its comment):

```toml
# genoray 2.9.0 adds is_symbolic / is_breakend filters used by the hap-safe SVAR
# conversion (benchmark_svar_convert.py). Pin explicitly (was transitive).
genoray = "==2.9.0"
```

- [ ] **Step 2: Add explicit genoray pin to `bench` (default)**

Under `[feature.bench.pypi-dependencies]`, add after the `genvarloader` line:

```toml
# Match bench027's genoray so bench_svar_vcf_plink.py imports a consistent API.
genoray = "==2.9.0"
```

- [ ] **Step 3: Resolve the lock**

Run: `pixi install -e bench027 && pixi install -e bench`
Expected: both environments solve; `pixi.lock` updates with `genoray 2.9.0` in both. No resolver error.

- [ ] **Step 4: Verify the pinned version imports in both envs**

Run:
```bash
pixi run -e bench027 python -c "import genoray; assert genoray.__version__ == '2.9.0', genoray.__version__; print('bench027 OK', genoray.__version__)"
pixi run -e bench python -c "import genoray; assert genoray.__version__ == '2.9.0', genoray.__version__; print('bench OK', genoray.__version__)"
```
Expected: `bench027 OK 2.9.0` and `bench OK 2.9.0`.

- [ ] **Step 5: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "build: pin genoray ==2.9.0 in bench027 and bench envs"
```

---

## Task 2: Shared `_bench_common.py` (byte accounting + measurement loop)

**Files:**
- Create: `hap_track_throughput/bin/_bench_common.py`
- Create: `hap_track_throughput/bin/tests/conftest.py`
- Create: `hap_track_throughput/bin/tests/test_bench_common.py`

This module is the single source of truth for: total-bytes-per-batch, MiB/s, and the timed measurement loop **with the empty-epoch guard**. Both `benchmark_haps.py` and `benchmark_tracks.py` import it (DRY). The measurement loop is extracted into a pure function so it is unit-testable with fake dataloaders and a fake clock — no genvarloader, torch, or real timing needed in tests.

- [ ] **Step 1: Write the conftest so pytest can import `bin/` modules**

Create `hap_track_throughput/bin/tests/conftest.py`:

```python
"""Put the parent `bin/` dir on sys.path so tests import sibling modules by bare
name (`_bench_common`, `_genoray_filter`) exactly as Nextflow does on PATH."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 2: Write the failing tests**

Create `hap_track_throughput/bin/tests/test_bench_common.py`:

```python
import numpy as np
import pytest

from _bench_common import (
    MEMORY_HEADER,
    THROUGHPUT_HEADER,
    CellResult,
    mib_per_s,
    n_bytes,
    measure_cell,
)


def test_headers_match_reconciled_schema():
    assert THROUGHPUT_HEADER == (
        "dataset,backend,dl_mode,threads,seqlen,batch_size,"
        "n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)\n"
    )
    assert MEMORY_HEADER == (
        "dataset,backend,dl_mode,threads,seqlen,batch_size,"
        "avg_rss_bytes,peak_rss_bytes\n"
    )


def test_n_bytes_numpy():
    arr = np.zeros((4, 2048), dtype="S1")
    assert n_bytes(arr) == 4 * 2048 * 1
    farr = np.zeros((4, 2048), dtype=np.float32)
    assert n_bytes(farr) == 4 * 2048 * 4


def test_n_bytes_torch_like():
    # torch.Tensor exposes numel()/element_size() and ALSO a .size method +
    # .itemsize attr; the numpy-first dispatch must take the torch branch.
    class FakeTensor:
        itemsize = 4

        def numel(self):
            return 8

        def element_size(self):
            return 4

        def size(self):
            return (2, 4)

    assert n_bytes(FakeTensor()) == 8 * 4


def test_mib_per_s():
    assert mib_per_s(2**20, 1.0) == pytest.approx(1.0)


class _FakeBatch:
    """Minimal numpy-like: n_bytes uses .size and .itemsize."""

    def __init__(self, nbytes: int):
        self.size = nbytes
        self.itemsize = 1


def _clock():
    """Deterministic ns clock advancing 1 ns per call."""
    t = {"n": 0}

    def now() -> int:
        t["n"] += 1
        return t["n"]

    return now


def test_measure_cell_counts_bytes_after_burn_in():
    # 3 batches of 100 bytes; burn_in=1 -> bytes counted for batches with
    # n_yielded >= burn_in (i.e. all 3 here, since burn_in index == first batch).
    dl = [_FakeBatch(100), _FakeBatch(100), _FakeBatch(100)]
    res = measure_cell(
        dl, burn_in=1, n_batches=3, time_limit_ns=10**18, min_batches=1,
        now_ns=_clock(),
    )
    assert isinstance(res, CellResult)
    # batches with n_yielded >= burn_in(=1): all three (n_yielded 1,2,3)
    assert res.total_bytes == 300
    # measured = batches with n_yielded > burn_in handling: see impl; here 3 counted
    assert res.n_measured == 3
    assert res.duration_ns > 0


def test_measure_cell_empty_epoch_returns_none():
    # An epoch that yields zero batches must NOT spin forever; returns None (NaN).
    res = measure_cell(
        [], burn_in=1, n_batches=5, time_limit_ns=10**18, min_batches=1,
        now_ns=_clock(),
    )
    assert res is None


def test_measure_cell_reiterates_until_n_batches():
    # A 2-batch loader, asked for 4 measured batches, must re-iterate (epoch twice).
    dl = [_FakeBatch(10), _FakeBatch(10)]
    res = measure_cell(
        dl, burn_in=0, n_batches=4, time_limit_ns=10**18, min_batches=1,
        now_ns=_clock(),
    )
    assert res is not None
    assert res.n_measured == 4
    assert res.total_bytes == 40
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd hap_track_throughput/bin && pixi run -e bench027 python -m pytest tests/test_bench_common.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_bench_common'` (module not created yet).

- [ ] **Step 4: Write the implementation**

Create `hap_track_throughput/bin/_bench_common.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd hap_track_throughput/bin && pixi run -e bench027 python -m pytest tests/test_bench_common.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add hap_track_throughput/bin/_bench_common.py hap_track_throughput/bin/tests/conftest.py hap_track_throughput/bin/tests/test_bench_common.py
git commit -m "feat(bench): shared byte-accounting + measurement loop with empty-epoch guard"
```

---

## Task 3: Rewrite `benchmark_haps.py` (dl_mode + reconciled schema)

**Files:**
- Modify: `hap_track_throughput/bin/benchmark_haps.py` (full rewrite)

The script now takes `--dl-mode {none,buffered}`, accumulates real batch bytes via `_bench_common.measure_cell`, and emits the reconciled throughput schema (`...,n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)`) or the dl_mode-augmented memory schema. Buffered haps requires `deterministic=True`; buffered construction `ValueError` and empty epochs both record NaN and continue.

This task has no cheap unit test of its own (it drives genvarloader on a real dataset); its core logic — `measure_cell` and the byte/MiB helpers — is already covered by Task 2. Verification is the Task 10 smoke run.

- [ ] **Step 1: Replace the file contents**

Overwrite `hap_track_throughput/bin/benchmark_haps.py` with:

```python
#! /usr/bin/env python
"""Haplotype dataloader throughput / memory sweep for the full GVL 0.27.0 bench.

Sweeps a (threads, batch_size, n_batches) grid for one dataset + seqlen, in one
dataloader mode (none|buffered). Throughput mode emits bytes-accounted MiB/s
(directly comparable to results/hap_results.csv); memory mode emits peak/avg RSS.
"""

from pathlib import Path

from cyclopts import run


def bench(
    results: Path,
    ds_path: Path,
    length: int,
    fasta: Path,
    grid_file: Path,
    dataset: str = "",
    backend: str = "svar",
    dl_mode: str = "none",  # "none" or "buffered"
    buffer_bytes: int = 2 * 2**30,
    burn_in: int = 1,
    replicates: int = 3,
    measure_memory: bool = False,
    time_limit_s: float = 45.0,
    min_batches: int = 5,
):
    import gc
    import os
    from itertools import product
    from time import sleep

    import genvarloader as gvl
    import numba as nb
    import polars as pl

    from _bench_common import (
        MEMORY_HEADER,
        THROUGHPUT_HEADER,
        measure_cell,
        mib_per_s,
    )

    if dl_mode not in ("none", "buffered"):
        raise ValueError(f"dl_mode must be 'none' or 'buffered', got {dl_mode!r}")

    ds = (
        gvl.Dataset.open(ds_path, fasta)
        .with_tracks(False)
        .with_seqs("haplotypes")
        .with_len(length)
        .with_settings(deterministic=True)  # required by buffered haps; harmless for none
    )
    dataset = dataset or ds_path.parent.name

    max_threads = len(os.sched_getaffinity(0))
    grid = pl.read_csv(grid_file)
    assert int(grid["threads"].max()) <= max_threads  # type: ignore

    time_limit_ns = int(time_limit_s * 1e9)
    dl_kwargs = {} if dl_mode == "none" else {"mode": "buffered", "buffer_bytes": buffer_bytes}

    if measure_memory:
        from _mem_sampler import PeakRssSampler

        with open(results, "w") as f:
            f.write(MEMORY_HEADER)
            f.flush()
            for (n_thread, batch_size, n_batches), _ in product(grid.iter_rows(), range(replicates)):
                nb.set_num_threads(n_thread)
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **dl_kwargs)
                except ValueError as e:
                    print(f"SKIP mem t={n_thread} bs={batch_size} ({dl_mode}): {e}", flush=True)
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},nan,nan\n")
                    f.flush()
                    continue
                with PeakRssSampler() as s:
                    res = measure_cell(
                        dl, burn_in=burn_in, n_batches=n_batches,
                        time_limit_ns=time_limit_ns, min_batches=min_batches,
                    )
                del dl
                gc.collect()
                sleep(0.5)
                if res is None:  # empty epoch -> NaN
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},nan,nan\n")
                else:
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},{s.avg},{s.peak}\n")
                f.flush()
    else:
        with open(results, "w") as f:
            f.write(THROUGHPUT_HEADER)
            f.flush()
            for (n_thread, batch_size, n_batches), _ in product(grid.iter_rows(), range(replicates)):
                nb.set_num_threads(n_thread)
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **dl_kwargs)
                except ValueError as e:
                    print(f"SKIP cell t={n_thread} bs={batch_size} ({dl_mode}): {e}", flush=True)
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},0,0,0,nan\n")
                    f.flush()
                    continue
                res = measure_cell(
                    dl, burn_in=burn_in, n_batches=n_batches,
                    time_limit_ns=time_limit_ns, min_batches=min_batches,
                )
                del dl
                gc.collect()
                if res is None:  # empty epoch -> NaN
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},0,0,0,nan\n")
                else:
                    tput = (
                        mib_per_s(res.total_bytes, res.duration_ns / 1e9)
                        if res.duration_ns > 0
                        else float("nan")
                    )
                    f.write(
                        f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},"
                        f"{res.n_measured},{res.total_bytes},{res.duration_ns},{tput}\n"
                    )
                f.flush()


if __name__ == "__main__":
    run(bench)
```

- [ ] **Step 2: Verify it imports and shows help under bench027**

Run: `cd hap_track_throughput/bin && pixi run -e bench027 python benchmark_haps.py --help`
Expected: cyclopts help text listing `--dl-mode`, `--measure-memory`, `--buffer-bytes` etc., no import error.

- [ ] **Step 3: Commit**

```bash
git add hap_track_throughput/bin/benchmark_haps.py
git commit -m "feat(bench): haps dl_mode + bytes-accounted MiB/s schema"
```

---

## Task 4: Rewrite `benchmark_tracks.py` (dl_mode + reconciled schema)

**Files:**
- Modify: `hap_track_throughput/bin/benchmark_tracks.py` (full rewrite)

Same shape as Task 3, for tracks. Tracks use `with_seqs(None).with_tracks("read-depth", "tracks")` and have **no** determinism requirement, so we don't set `deterministic=True`. Buffered tracks still honor the `ValueError`→NaN and empty-epoch→NaN guards.

- [ ] **Step 1: Replace the file contents**

Overwrite `hap_track_throughput/bin/benchmark_tracks.py` with:

```python
#! /usr/bin/env python
"""Track dataloader throughput / memory sweep for the full GVL 0.27.0 bench.

Sweeps a (threads, batch_size, n_batches) grid for one dataset + seqlen, in one
dataloader mode (none|buffered). Throughput mode emits bytes-accounted MiB/s
(directly comparable to results/track_results.csv); memory mode emits peak/avg RSS.
"""

from pathlib import Path

from cyclopts import run


def bench(
    results: Path,
    ds_path: Path,
    length: int,
    fasta: Path,
    grid_file: Path,
    dataset: str = "",
    backend: str = "svar",
    dl_mode: str = "none",  # "none" or "buffered"
    buffer_bytes: int = 2 * 2**30,
    burn_in: int = 1,
    replicates: int = 3,
    measure_memory: bool = False,
    time_limit_s: float = 45.0,
    min_batches: int = 5,
):
    import gc
    import os
    from itertools import product
    from time import sleep

    import genvarloader as gvl
    import numba as nb
    import polars as pl

    from _bench_common import (
        MEMORY_HEADER,
        THROUGHPUT_HEADER,
        measure_cell,
        mib_per_s,
    )

    if dl_mode not in ("none", "buffered"):
        raise ValueError(f"dl_mode must be 'none' or 'buffered', got {dl_mode!r}")

    ds = (
        gvl.Dataset.open(ds_path, fasta)
        .with_seqs(None)
        .with_tracks("read-depth", "tracks")
        .with_len(length)
    )
    dataset = dataset or ds_path.parent.name

    max_threads = len(os.sched_getaffinity(0))
    grid = pl.read_csv(grid_file)
    assert int(grid["threads"].max()) <= max_threads  # type: ignore

    time_limit_ns = int(time_limit_s * 1e9)
    dl_kwargs = {} if dl_mode == "none" else {"mode": "buffered", "buffer_bytes": buffer_bytes}

    if measure_memory:
        from _mem_sampler import PeakRssSampler

        with open(results, "w") as f:
            f.write(MEMORY_HEADER)
            f.flush()
            for (n_thread, batch_size, n_batches), _ in product(grid.iter_rows(), range(replicates)):
                nb.set_num_threads(n_thread)
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **dl_kwargs)
                except ValueError as e:
                    print(f"SKIP mem t={n_thread} bs={batch_size} ({dl_mode}): {e}", flush=True)
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},nan,nan\n")
                    f.flush()
                    continue
                with PeakRssSampler() as s:
                    res = measure_cell(
                        dl, burn_in=burn_in, n_batches=n_batches,
                        time_limit_ns=time_limit_ns, min_batches=min_batches,
                    )
                del dl
                gc.collect()
                sleep(0.5)
                if res is None:
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},nan,nan\n")
                else:
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},{s.avg},{s.peak}\n")
                f.flush()
    else:
        with open(results, "w") as f:
            f.write(THROUGHPUT_HEADER)
            f.flush()
            for (n_thread, batch_size, n_batches), _ in product(grid.iter_rows(), range(replicates)):
                nb.set_num_threads(n_thread)
                try:
                    dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **dl_kwargs)
                except ValueError as e:
                    print(f"SKIP cell t={n_thread} bs={batch_size} ({dl_mode}): {e}", flush=True)
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},0,0,0,nan\n")
                    f.flush()
                    continue
                res = measure_cell(
                    dl, burn_in=burn_in, n_batches=n_batches,
                    time_limit_ns=time_limit_ns, min_batches=min_batches,
                )
                del dl
                gc.collect()
                if res is None:
                    f.write(f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},0,0,0,nan\n")
                else:
                    tput = (
                        mib_per_s(res.total_bytes, res.duration_ns / 1e9)
                        if res.duration_ns > 0
                        else float("nan")
                    )
                    f.write(
                        f"{dataset},{backend},{dl_mode},{n_thread},{length},{batch_size},"
                        f"{res.n_measured},{res.total_bytes},{res.duration_ns},{tput}\n"
                    )
                f.flush()


if __name__ == "__main__":
    run(bench)
```

- [ ] **Step 2: Verify it imports and shows help**

Run: `cd hap_track_throughput/bin && pixi run -e bench027 python benchmark_tracks.py --help`
Expected: cyclopts help with `--dl-mode`, no import error.

- [ ] **Step 3: Commit**

```bash
git add hap_track_throughput/bin/benchmark_tracks.py
git commit -m "feat(bench): tracks dl_mode + bytes-accounted MiB/s schema"
```

---

## Task 5: Hap-safe genoray filter at SVAR conversion

**Files:**
- Create: `hap_track_throughput/bin/_genoray_filter.py`
- Create: `hap_track_throughput/bin/tests/test_genoray_filter.py`
- Modify: `hap_track_throughput/bin/benchmark_svar_convert.py`

genoray 2.9.0's `~is_symbolic & ~is_breakend` drops every ALT a haplotype consumer can't expand. PGEN takes the polars expr only; VCF/BCF requires both a cyvcf2 callable **and** the polars expr. We centralize the builder in `_genoray_filter.py` (public-API only — no `import polars`, no genoray internals) so the SVAR-convert and the `bench_svar_vcf_plink.py` fix share one definition.

- [ ] **Step 1: Write the failing tests**

Create `hap_track_throughput/bin/tests/test_genoray_filter.py`:

```python
import polars as pl
import pytest

from _genoray_filter import hap_safe_pl_filter, hap_safe_vcf_callable


@pytest.mark.parametrize(
    "alt,expected_keep",
    [
        (["A"], True),
        (["AT"], True),
        (["<DEL>"], False),
        (["G[chr2:321["], False),
        ([".TGCA"], False),
        (["TGCA."], False),
    ],
)
def test_pl_filter_keeps_only_expandable_alts(alt, expected_keep):
    # is_symbolic / is_breakend operate on the ALT list column.
    df = pl.DataFrame({"ALT": [alt]})
    kept = df.filter(hap_safe_pl_filter())
    assert (kept.height == 1) is expected_keep


@pytest.mark.parametrize(
    "alts,expected_keep",
    [
        (["A"], True),
        (["AT"], True),
        (["<DEL>"], False),
        (["G[chr2:321["], False),
        ([".TGCA"], False),
        (["TGCA."], False),
        (["A", "<INS>"], False),  # any disallowed ALT drops the record
    ],
)
def test_vcf_callable_matches_pl_filter(alts, expected_keep):
    keep = hap_safe_vcf_callable()
    assert keep(alts) is expected_keep


def test_flags_select_subset():
    # no_symbolic only: breakend ALT survives the pl_filter
    df = pl.DataFrame({"ALT": [["G[chr2:321["]]})
    assert df.filter(hap_safe_pl_filter(no_symbolic=True, no_breakend=False)).height == 1
    # no_breakend only: symbolic ALT survives
    df2 = pl.DataFrame({"ALT": [["<DEL>"]]})
    assert df2.filter(hap_safe_pl_filter(no_symbolic=False, no_breakend=True)).height == 1
    # neither flag: filter is a no-op (keeps everything)
    df3 = pl.DataFrame({"ALT": [["<DEL>"], ["G[chr2:321["]]})
    assert df3.filter(hap_safe_pl_filter(no_symbolic=False, no_breakend=False)).height == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd hap_track_throughput/bin && pixi run -e bench027 python -m pytest tests/test_genoray_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_genoray_filter'`.

- [ ] **Step 3: Write the implementation**

Create `hap_track_throughput/bin/_genoray_filter.py`:

```python
"""Hap-safe variant filter shared by the SVAR conversion + the VCF/PLINK bench.

genvarloader cannot expand symbolic (`<DEL>`) or breakend (`G[chr2:321[`) ALT
alleles into literal nucleotides, so both must be dropped before any variant
reaches haplotype buffers. genoray 2.9.0 ships the polars filter expressions
`genoray.exprs.is_symbolic` / `is_breakend` (public). PGEN readers take the
polars expr alone; VCF/BCF readers require BOTH a cyvcf2 callable and the
matching polars expr (genoray enforces the both-or-neither invariant).

Public-API only: combining genoray.exprs with `&`/`~` needs no `import polars`,
and the breakend regex is copied from genoray.exprs._BND_PATTERN's documented
form (kept in sync by the unit tests in tests/test_genoray_filter.py).
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

# Mirror of genoray.exprs._BND_PATTERN (VCF 4.x breakend ALT replacement string).
# Matches mate-pair forms (contain `[` or `]`) and single-breakend forms
# (a base adjacent to a `.`). A lone `.` (no-ALT) does not match.
_BND_PATTERN = r"[\[\]]|^\.[A-Za-z]|[A-Za-z]\.$"


def hap_safe_pl_filter(no_symbolic: bool = True, no_breakend: bool = True):
    """Polars filter expression keeping only haplotype-expandable variants.

    Returns a `pl.Expr`. With both flags False, returns an all-True no-op expr.
    """
    import genoray

    expr = None
    if no_symbolic:
        expr = ~genoray.exprs.is_symbolic
    if no_breakend:
        be = ~genoray.exprs.is_breakend
        expr = be if expr is None else (expr & be)
    if expr is None:
        import polars as pl

        return pl.lit(True)
    return expr


def hap_safe_vcf_callable(
    no_symbolic: bool = True, no_breakend: bool = True
) -> Callable[[Iterable[str]], bool]:
    """cyvcf2-style callable mirroring `hap_safe_pl_filter`.

    Accepts an iterable of ALT strings (a `cyvcf2.Variant.ALT`) and returns True
    to KEEP the record. Pass directly as `VCF(filter=...)` alongside
    `pl_filter=hap_safe_pl_filter(...)`.
    """

    def keep(alts: Iterable[str]) -> bool:
        alts = list(alts)
        if no_symbolic and any(a.startswith("<") for a in alts):
            return False
        if no_breakend and any(re.search(_BND_PATTERN, a) is not None for a in alts):
            return False
        return True

    return keep


def open_filtered_reader(variants, no_symbolic: bool = True, no_breakend: bool = True):
    """Open a genoray PGEN/VCF reader with the hap-safe filter applied.

    Dispatches on suffix: `.pgen` -> PGEN(filter=expr); `.bcf`/`.vcf`/`.vcf.gz`
    -> VCF(filter=callable, pl_filter=expr). Returns (reader, source_fmt).
    """
    import genoray

    name = str(variants).lower()
    pl_filter = hap_safe_pl_filter(no_symbolic, no_breakend)
    if name.endswith(".pgen"):
        return genoray.PGEN(variants, filter=pl_filter), "pgen"
    if name.endswith(".bcf") or name.endswith(".vcf") or name.endswith(".vcf.gz"):
        cb = hap_safe_vcf_callable(no_symbolic, no_breakend)
        source_fmt = "bcf" if name.endswith(".bcf") else "vcf"
        return genoray.VCF(variants, filter=cb, pl_filter=pl_filter), source_fmt
    raise ValueError(f"Unsupported variant format: {variants}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd hap_track_throughput/bin && pixi run -e bench027 python -m pytest tests/test_genoray_filter.py -v`
Expected: all PASS.

- [ ] **Step 5: Wire the filter into `benchmark_svar_convert.py`**

Overwrite `hap_track_throughput/bin/benchmark_svar_convert.py` with:

```python
#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def bench(
    results: Path,
    variants: Path,
    out_svar: Path,
    dataset: str = "",
    measure_memory: bool = False,
    max_mem: str = "4g",
    n_jobs: int = -1,
    no_symbolic: bool = True,
    no_breakend: bool = True,
):
    from time import perf_counter_ns

    import genoray

    from _genoray_filter import open_filtered_reader

    source, source_fmt = open_filtered_reader(variants, no_symbolic, no_breakend)

    if source_fmt == "pgen":

        def convert():
            genoray.SparseVar.from_pgen(out_svar, source, max_mem, overwrite=True, n_jobs=n_jobs)

    else:  # bcf / vcf

        def convert():
            genoray.SparseVar.from_vcf(out_svar, source, max_mem, overwrite=True, n_jobs=n_jobs)

    dataset = dataset or variants.stem

    if measure_memory:
        from _mem_sampler import PeakRssSampler

        with PeakRssSampler() as s:
            convert()
        with open(results, "w") as f:
            f.write("dataset,source_fmt,n_jobs,avg_rss_bytes,peak_rss_bytes\n")
            f.write(f"{dataset},{source_fmt},{n_jobs},{s.avg},{s.peak}\n")
    else:
        t0 = perf_counter_ns()
        convert()
        duration = perf_counter_ns() - t0
        with open(results, "w") as f:
            f.write("dataset,source_fmt,n_jobs,duration\n")
            f.write(f"{dataset},{source_fmt},{n_jobs},{duration}\n")


if __name__ == "__main__":
    run(bench)
```

- [ ] **Step 6: Verify the convert script imports and parses flags**

Run: `cd hap_track_throughput/bin && pixi run -e bench027 python benchmark_svar_convert.py --help`
Expected: help text shows `--no-symbolic` / `--no-breakend` (and their `--no-no-symbolic`/negation forms per cyclopts), no import error.

- [ ] **Step 7: Commit**

```bash
git add hap_track_throughput/bin/_genoray_filter.py hap_track_throughput/bin/tests/test_genoray_filter.py hap_track_throughput/bin/benchmark_svar_convert.py
git commit -m "feat(bench): hap-safe genoray symbolic/breakend filter at SVAR conversion"
```

---

## Task 6: Correctness fix to `bench_svar_vcf_plink.py` (not re-run)

**Files:**
- Modify: `hap_track_throughput/bench_svar_vcf_plink.py:314,320-322`

`bench_svar_vcf_plink.py` is **not re-executed** in this project — it only needs to stay importable and correct under genoray 2.9.0. Apply the same hap-safe filter to its `VCF(...)` and `PGEN(...)` construction. It lives one level up from `bin/`, so import the shared helper by adding `bin/` to `sys.path` at the top.

- [ ] **Step 1: Add the shared-helper import**

In `hap_track_throughput/bench_svar_vcf_plink.py`, the top imports currently are (lines 4-16):

```python
import os
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import awkward as ak
import numpy as np
from awkward.contents import Content
from numba import njit, prange
from numpy.typing import NDArray
```

Replace the `import os` line with these two lines (adds `bin/` to the path so `_genoray_filter` resolves):

```python
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "bin"))
```

- [ ] **Step 2: Filter the readers in `main`**

In `main` (the body starts at line 313 `import polars as pl`), the reader construction is:

```python
    _svar = SparseVar(svar)
    _bcf = VCF(bcf, with_gvi_index=False)
    _pgen = PGEN(pgen)
```

Replace those three lines with:

```python
    from _genoray_filter import hap_safe_pl_filter, hap_safe_vcf_callable

    _hap_pl = hap_safe_pl_filter()
    _svar = SparseVar(svar)
    _bcf = VCF(bcf, with_gvi_index=False, filter=hap_safe_vcf_callable(), pl_filter=_hap_pl)
    _pgen = PGEN(pgen, filter=_hap_pl)
```

> Note: `with_gvi_index=False` is preserved (the bench reads the BCF without building a `.gvi`). The `filter`/`pl_filter` pair satisfies genoray's both-or-neither invariant.

- [ ] **Step 3: Verify the module still imports under `bench` (default env)**

Run: `cd hap_track_throughput && pixi run -e bench python -c "import ast,sys; ast.parse(open('bench_svar_vcf_plink.py').read()); print('parse OK')"`
Then a real import smoke (no execution of `main`):
Run: `cd hap_track_throughput && pixi run -e bench python -c "import sys; sys.argv=['x']; import importlib.util as u; spec=u.spec_from_file_location('bsvp','bench_svar_vcf_plink.py'); m=u.module_from_spec(spec); spec.loader.exec_module(m); print('import OK', hasattr(m,'main'))"`
Expected: `parse OK` then `import OK True`. (The `if __name__ == '__main__'` block does not run on import, so no benchmark executes.)

- [ ] **Step 4: Commit**

```bash
git add hap_track_throughput/bench_svar_vcf_plink.py
git commit -m "fix(bench): apply hap-safe genoray filter in bench_svar_vcf_plink (correctness only)"
```

---

## Task 7: `benchmark.nf` — SVAR-only, dl_mode dimension, filter params

**Files:**
- Modify: `hap_track_throughput/benchmark.nf`

Add `bench_native` (default false) to gate the native write branch off, so every dataset is SVAR-backed. Add `no_symbolic`/`no_breakend` params threaded into `BENCH_SVAR_CONVERT`. Cross a new `dl_modes` channel into the haps/track inputs so each `(length, backend, dl_mode)` is a separate process + CSV. Point `results_dir` at the repo-level `results_gvl027/`. This is typed strict-syntax Nextflow (26.04) — follow the typed-nextflow conventions already used in this file (records, `output {}`, typed `params {}`).

- [ ] **Step 1: Extend the `params {}` block**

Replace the existing `params { ... }` block (lines 7-21) with:

```nextflow
params {
    dataset: String
    fasta: Path
    variants: List<Path>
    ds_dir: String
    bigwig_table: Path?
    region: String?
    min_npb: Integer? = null
    max_npb: Integer = 2 ** 33
    test_grid: Boolean = false
    bench_haps: Boolean = true
    bench_tracks: Boolean = false
    bench_native: Boolean = false
    no_symbolic: Boolean = true
    no_breakend: Boolean = true
    measure_memory: Boolean = false
    results_dir: String = "${projectDir}/../results_gvl027"
}
```

- [ ] **Step 2: Make writes SVAR-only and cross in `dl_mode`**

Replace the workflow `main:` body from the `lengths = ...` line through the `tracks_results = ...` line (lines 36-56) with:

```nextflow
    lengths = channel.fromList([2048, 16384, 131072, 1048576])
    dl_modes = channel.fromList(["none", "buffered"])

    // SVAR conversion bench (runs once, not per seqlen). Hap-safe filtering
    // (drop symbolic + breakend ALTs) is applied here so every SVAR is
    // GVL-compatible.
    svar_result = BENCH_SVAR_CONVERT(params.dataset, params.variants)

    // SVAR write inputs: cartesian product of lengths x the single converted svar.
    svar_write_inputs = lengths
        .combine(svar_result.map { r -> r.svar })
        .map { len, svar -> record(length: len, variants: [svar] as List<Path>, backend: "svar") }

    // Native write inputs (raw variants, no SVAR) — gated off by default.
    native_write_inputs = params.bench_native
        ? lengths.map { len -> record(length: len, variants: params.variants, backend: "native") }
        : channel.empty()

    all_write_inputs = svar_write_inputs.mix(native_write_inputs)

    // Dataset write bench (per length x backend)
    write_results = BENCH_WRITE_DATASET(all_write_inputs, params.bigwig_table, params.region)

    // Cross each written dataset with both dataloader modes. The same .gvl is
    // reused across dl_modes (write does not depend on dl_mode).
    dl_inputs = write_results
        .combine(dl_modes)
        .map { wr, dl ->
            record(length: wr.length, backend: wr.backend, gvl: wr.gvl, dl_mode: dl)
        }

    haps_results = params.bench_haps ? BENCH_HAPS(dl_inputs) : channel.empty()
    do_tracks = params.bench_tracks && params.bigwig_table != null
    tracks_results = do_tracks ? BENCH_TRACKS(dl_inputs) : channel.empty()
```

> Note: `combine` is still a core operator under static types and pairs every element of the left channel with every element of the right (cartesian). `write_results.combine(dl_modes)` yields `(BenchWriteResult, String)` pairs.

- [ ] **Step 3: Add `_${dl_mode}` to the haps/tracks output paths**

Replace the `haps_results` and `tracks_results` blocks inside `output { ... }` (lines 77-86) with:

```nextflow
    haps_results: Channel<BenchResult> {
        path { r ->
            r.csv >> "${params.results_dir}/${params.measure_memory ? 'haps_memory' : 'haps'}/${params.dataset}_${r.length}_${r.backend}_${r.dl_mode}.csv"
        }
    }
    tracks_results: Channel<BenchResult> {
        path { r ->
            r.csv >> "${params.results_dir}/${params.measure_memory ? 'tracks_memory' : 'tracks'}/${params.dataset}_${r.length}_${r.backend}_${r.dl_mode}.csv"
        }
    }
```

- [ ] **Step 4: Pass the filter flags into `BENCH_SVAR_CONVERT`**

In `process BENCH_SVAR_CONVERT`, replace the `script:` section (lines 99-111) with:

```nextflow
    script:
    vars = variants.first()
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    sym_flag = params.no_symbolic ? "--no-symbolic" : "--no-no-symbolic"
    bnd_flag = params.no_breakend ? "--no-breakend" : "--no-no-breakend"
    """
    benchmark_svar_convert.py \\
      svar_convert.csv \\
      ${vars} \\
      output.svar \\
      --dataset ${dataset} \\
      --max-mem 64g \\
      --n-jobs ${task.cpus} \\
      ${sym_flag} \\
      ${bnd_flag} \\
      ${mem_flag}
    """
```

> `cyclopts` renders boolean `no_symbolic: bool = True` as `--no-symbolic` (set True) / `--no-no-symbolic` (set False). The flags are always passed so the value is explicit regardless of the default.

- [ ] **Step 5: Thread `dl_mode` through `BENCH_HAPS`**

In `process BENCH_HAPS`, replace the `benchmark_haps.py` invocation inside `script:` (lines 188-196) so it passes `--dl-mode` and the dataset name, and rename the output CSV to include dl_mode. Replace the whole `script:` block (lines 174-197) with:

```nextflow
    script:
    min_npb_arg = params.min_npb != null ? "--min-npb ${params.min_npb}" : ""
    test_arg = params.test_grid ? "--test" : ""
    mem_grid_arg = params.measure_memory ? "--memory-grid" : ""
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    """
    make_launch_grid.py \\
      ${ds.length} \\
      --max-npb ${params.max_npb} \\
      ${min_npb_arg} \\
      ${test_arg} \\
      ${mem_grid_arg} \\
      --output grid_${ds.length}.csv

    benchmark_haps.py \\
      results_${ds.length}_${ds.backend}_${ds.dl_mode}.csv \\
      ${ds.gvl} \\
      ${ds.length} \\
      ${params.fasta} \\
      grid_${ds.length}.csv \\
      --dataset ${params.dataset} \\
      --backend ${ds.backend} \\
      --dl-mode ${ds.dl_mode} \\
      ${mem_flag}
    """
```

Then replace the `BENCH_HAPS` `output:` block (lines 199-200) with:

```nextflow
    output:
    record(length: ds.length, backend: ds.backend, dl_mode: ds.dl_mode, csv: file("results_${ds.length}_${ds.backend}_${ds.dl_mode}.csv"))
```

- [ ] **Step 6: Thread `dl_mode` through `BENCH_TRACKS`**

In `process BENCH_TRACKS`, replace the `script:` block (lines 214-237) with:

```nextflow
    script:
    min_npb_arg = params.min_npb != null ? "--min-npb ${params.min_npb}" : ""
    test_arg = params.test_grid ? "--test" : ""
    mem_grid_arg = params.measure_memory ? "--memory-grid" : ""
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    """
    make_launch_grid.py \\
      ${ds.length} \\
      --max-npb ${params.max_npb} \\
      ${min_npb_arg} \\
      ${test_arg} \\
      ${mem_grid_arg} \\
      --output grid_${ds.length}.csv

    benchmark_tracks.py \\
      results_${ds.length}_${ds.backend}_${ds.dl_mode}.csv \\
      ${ds.gvl} \\
      ${ds.length} \\
      ${params.fasta} \\
      grid_${ds.length}.csv \\
      --dataset ${params.dataset} \\
      --backend ${ds.backend} \\
      --dl-mode ${ds.dl_mode} \\
      ${mem_flag}
    """
```

Then replace the `BENCH_TRACKS` `output:` block (lines 239-240) with:

```nextflow
    output:
    record(length: ds.length, backend: ds.backend, dl_mode: ds.dl_mode, csv: file("results_${ds.length}_${ds.backend}_${ds.dl_mode}.csv"))
```

- [ ] **Step 7: Add `dl_mode` to the `Dataset` and `BenchResult` records**

Replace the `record Dataset { ... }` block (lines 243-247) with:

```nextflow
record Dataset {
    length: Integer
    backend: String
    dl_mode: String
    gvl: Path
}
```

Replace the `record BenchResult { ... }` block (lines 255-259) with:

```nextflow
record BenchResult {
    length: Integer
    backend: String
    dl_mode: String
    csv: Path
}
```

> The `WriteInput`, `SvarConvertResult`, and `BenchWriteResult` records are unchanged.

- [ ] **Step 8: Lint the pipeline (parse + types, no execution)**

Run: `cd hap_track_throughput && nextflow lint benchmark.nf`
Expected: no errors. (If `nextflow lint` is unavailable on this install, fall back to `nextflow run benchmark.nf -profile gvl027 -c configs/1kgp.config -preview -stub` once Task 8 lands, and confirm the DAG builds without a syntax/type error.)

- [ ] **Step 9: Commit**

```bash
git add hap_track_throughput/benchmark.nf
git commit -m "feat(nf): SVAR-only writes, dl_mode dimension, hap-safe filter params, results_gvl027 output"
```

---

## Task 8: `nextflow.config` — `gvl027` profile

**Files:**
- Create: `hap_track_throughput/nextflow.config`

Each Nextflow process runs as a fresh SLURM job, so it must (a) target the SLURM executor and (b) activate the `bench027` pixi env before the script runs, so `python` / the `bin/` shebang scripts resolve to genvarloader 0.27.0. Nextflow itself is **not** added to pixi (conda Nextflow has Java-discovery issues) — the pipeline is launched with the user's local Nextflow, selecting `-profile gvl027`.

- [ ] **Step 1: Write the config**

Create `hap_track_throughput/nextflow.config`:

```groovy
// Profiles for the GVL full throughput + memory bench.
// Launch with the user's local Nextflow:
//   nextflow run benchmark.nf -profile gvl027 -c configs/<dataset>.config [--measure_memory]

profiles {
    gvl027 {
        process.executor = 'slurm'
        // Activate the bench027 pixi env on every compute node before the task
        // script runs, so `python` and the bin/ shebang scripts resolve to
        // genvarloader 0.27.0 / genoray 2.9.0. pixi must be on PATH first.
        process.beforeScript = '''
        export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
        eval "$(pixi shell-hook -e bench027 --manifest-path /carter/users/dlaub/projects/gvl-paper/pixi.toml)"
        # numba's thread-pool ceiling is fixed at import to the affinity count; the
        # grid asks for up to 64 threads, so make sure the ceiling permits it.
        export NUMBA_NUM_THREADS=64
        '''
    }
}

// Publish (workflow output {}) copies rather than symlinks so results survive
// work-dir cleanup.
workflow.output.mode = 'copy'
```

- [ ] **Step 2: Verify pixi shell-hook activates bench027 from a clean shell**

Run:
```bash
bash -lc 'export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"; eval "$(pixi shell-hook -e bench027 --manifest-path /carter/users/dlaub/projects/gvl-paper/pixi.toml)"; python -c "import genvarloader, genoray; print(genvarloader.__version__, genoray.__version__)"'
```
Expected: prints `0.27.0 2.9.0` (confirms the activation mechanism the `beforeScript` relies on works from a fresh shell). If `pixi shell-hook` errors, this is the place to discover it — not on a compute node.

- [ ] **Step 3: Commit**

```bash
git add hap_track_throughput/nextflow.config
git commit -m "feat(nf): gvl027 profile (slurm executor + bench027 env activation)"
```

---

## Task 9: Rewrite `compare_to_baseline.py` for the nf layout

**Files:**
- Modify: `hap_track_throughput/bin_gvl027/compare_to_baseline.py` (rewrite)
- Create: `hap_track_throughput/bin_gvl027/tests/test_compare_helpers.py`

The full-bench results land under `results_gvl027/{haps,tracks,haps_memory,tracks_memory}/{dataset}_{length}_{backend}_{dl_mode}.csv`. The output mode comes from the directory; `dl_mode` is both in the filename and a column. Dataset names are UPPERCASE in the new CSVs (`1KGP`) but lowercase/hyphenated in the baseline (`1kgp`), so the join normalizes both sides. Three reports: throughput internal parity (`dl_mode == "none"` joined to baseline), buffered throughput standalone, and memory absolute tables + a peak-RSS-vs-batch plot.

The pure helpers (dataset normalization, directory→mode parsing) are unit-tested; the plotting/aggregation is verified on the Task 10 smoke output.

- [ ] **Step 1: Write the failing helper tests**

Create `hap_track_throughput/bin_gvl027/tests/test_compare_helpers.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from compare_to_baseline import norm_dataset, parse_mode_dir


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1KGP", "1kgp"),
        ("TCGA_ATAC", "tcga-atac"),
        ("UKBB", "ukbb"),
        ("1kgp", "1kgp"),
        ("tcga-atac", "tcga-atac"),
    ],
)
def test_norm_dataset(raw, expected):
    assert norm_dataset(raw) == expected


@pytest.mark.parametrize(
    "dirname,expected",
    [
        ("haps", ("haps", False)),
        ("tracks", ("tracks", False)),
        ("haps_memory", ("haps", True)),
        ("tracks_memory", ("tracks", True)),
    ],
)
def test_parse_mode_dir(dirname, expected):
    assert parse_mode_dir(dirname) == expected


def test_parse_mode_dir_rejects_unknown():
    with pytest.raises(ValueError):
        parse_mode_dir("svar_convert")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd hap_track_throughput/bin_gvl027 && pixi run -e bench027 python -m pytest tests/test_compare_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'norm_dataset'` (functions not defined yet).

- [ ] **Step 3: Write the implementation**

Overwrite `hap_track_throughput/bin_gvl027/compare_to_baseline.py` with:

```python
#! /usr/bin/env python
"""Compare the GVL 0.27.0 full-bench results to the v0.6.1 baseline.

Reads the Nextflow output layout under results_gvl027/:
    haps/{dataset}_{length}_{backend}_{dl_mode}.csv          (throughput)
    tracks/{dataset}_{length}_{backend}_{dl_mode}.csv        (throughput)
    haps_memory/...   tracks_memory/...                      (peak/avg RSS)

Outputs:
  results_gvl027/parity_summary.csv  — throughput, dl_mode=="none" joined to the
       0.6.1 baseline on (dataset,mode,threads,seqlen,batch_size); ratio = v027/v061.
       INTERNAL sanity check, not a paper deliverable.
  results_gvl027/buffered_throughput.csv — buffered throughput, standalone (no baseline).
  results_gvl027/memory_summary.csv  — per (dataset,mode,dl_mode,seqlen,batch_size)
       peak/avg RSS, absolute (no baseline join).
  figures/gvl027_parity.png          — v061-vs-v027 throughput scatter (none mode).
  figures/gvl027_peak_rss.png        — peak RSS vs batch_size, faceted by seqlen.
"""

from pathlib import Path

import cyclopts


def norm_dataset(name: str) -> str:
    """Normalize a dataset label to the baseline convention (lower, '_'->'-')."""
    return name.lower().replace("_", "-")


def parse_mode_dir(dirname: str) -> tuple[str, bool]:
    """Map an nf output subdir to (output_mode, is_memory).

    'haps'->('haps',False); 'haps_memory'->('haps',True); same for 'tracks'.
    Raises ValueError for anything else (e.g. 'svar_convert', 'write').
    """
    if dirname in ("haps", "tracks"):
        return dirname, False
    if dirname in ("haps_memory", "tracks_memory"):
        return dirname[: -len("_memory")], True
    raise ValueError(f"not a haps/tracks result dir: {dirname!r}")


def _load_dir(results_dir: Path, dirname: str):
    """Concat all CSVs in results_dir/dirname, tagging output mode + dataset norm."""
    import polars as pl

    mode, _is_mem = parse_mode_dir(dirname)
    d = results_dir / dirname
    files = sorted(d.glob("*.csv"))
    if not files:
        return None
    df = pl.concat([pl.read_csv(p) for p in files], how="vertical_relaxed")
    return df.with_columns(
        mode=pl.lit(mode),
        dataset_norm=pl.col("dataset").map_elements(norm_dataset, return_dtype=pl.Utf8),
    )


def main(
    results_dir: Path = Path("results_gvl027"),
    hap_baseline: Path = Path("results/hap_results.csv"),
    track_baseline: Path = Path("results/track_results.csv"),
    fig_dir: Path = Path("figures"),
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (imported for side effects / future use)
    import polars as pl
    import seaborn as sns

    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # ---------- throughput ----------
    tput_frames = [
        f for f in (_load_dir(results_dir, d) for d in ("haps", "tracks")) if f is not None
    ]
    if tput_frames:
        tput = pl.concat(tput_frames, how="vertical_relaxed").rename(
            {"throughput (MiB/s)": "v027"}
        )
        # NaN cells (buffered skip / empty epoch) come in as numeric NaN or "nan".
        tput = (
            tput.with_columns(pl.col("v027").cast(pl.Float64, strict=False).fill_nan(None))
            .drop_nulls("v027")
            .group_by(["dataset_norm", "mode", "dl_mode", "threads", "seqlen", "batch_size"])
            .agg(pl.col("v027").median())
        )

        # baselines
        hap = pl.read_csv(hap_baseline).with_columns(mode=pl.lit("haps"))
        trk = pl.read_csv(track_baseline).with_columns(mode=pl.lit("tracks"))
        base = (
            pl.concat([hap, trk], how="vertical_relaxed")
            .rename({"throughput (MiB/s)": "v061"})
            .with_columns(
                dataset_norm=pl.col("dataset").map_elements(norm_dataset, return_dtype=pl.Utf8)
            )
            .group_by(["dataset_norm", "mode", "threads", "seqlen", "batch_size"])
            .agg(pl.col("v061").median())
        )

        # internal parity: dl_mode == none vs baseline
        none = tput.filter(pl.col("dl_mode") == "none")
        joined = none.join(
            base, on=["dataset_norm", "mode", "threads", "seqlen", "batch_size"], how="left"
        ).with_columns(ratio=(pl.col("v027") / pl.col("v061")))
        summary = results_dir / "parity_summary.csv"
        joined.sort(["dataset_norm", "mode", "seqlen", "batch_size", "threads"]).write_csv(summary)
        print(f"WROTE {summary} ({joined.height} rows)")

        matched = joined.drop_nulls("v061")
        if matched.height:
            print("Median 0.27.0/0.6.1 throughput ratio by (dataset, mode):")
            print(
                matched.group_by(["dataset_norm", "mode"])
                .agg(pl.col("ratio").median())
                .sort(["dataset_norm", "mode"])
            )
            pdf = matched.to_pandas()
            g = sns.relplot(
                data=pdf, x="v061", y="v027", hue="dataset_norm", style="mode",
                col="seqlen", col_wrap=2, facet_kws={"sharex": False, "sharey": False},
            )
            for ax in g.axes.flat:
                lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
                hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot([lo, hi], [lo, hi], ls="--", c="grey", lw=1)
                ax.set_xscale("log"); ax.set_yscale("log")
            g.set_axis_labels("v0.6.1 MiB/s", "v0.27.0 MiB/s")
            g.savefig(fig_dir / "gvl027_parity.png", dpi=150, bbox_inches="tight")
            print(f"WROTE {fig_dir / 'gvl027_parity.png'}")

        # buffered standalone
        buffered = tput.filter(pl.col("dl_mode") == "buffered")
        buf_out = results_dir / "buffered_throughput.csv"
        buffered.sort(["dataset_norm", "mode", "seqlen", "batch_size", "threads"]).write_csv(buf_out)
        print(f"WROTE {buf_out} ({buffered.height} rows)")
    else:
        print("No throughput CSVs found under haps/ or tracks/ — skipping throughput report.")

    # ---------- memory ----------
    mem_frames = [
        f
        for f in (_load_dir(results_dir, d) for d in ("haps_memory", "tracks_memory"))
        if f is not None
    ]
    if mem_frames:
        mem = pl.concat(mem_frames, how="vertical_relaxed")
        mem = (
            mem.with_columns(
                pl.col("peak_rss_bytes").cast(pl.Float64, strict=False).fill_nan(None),
                pl.col("avg_rss_bytes").cast(pl.Float64, strict=False).fill_nan(None),
            )
            .drop_nulls("peak_rss_bytes")
            .group_by(["dataset_norm", "mode", "dl_mode", "seqlen", "batch_size"])
            .agg(
                pl.col("peak_rss_bytes").max().alias("peak_rss_bytes"),
                pl.col("avg_rss_bytes").mean().alias("avg_rss_bytes"),
            )
        )
        mem_out = results_dir / "memory_summary.csv"
        mem.sort(["dataset_norm", "mode", "dl_mode", "seqlen", "batch_size"]).write_csv(mem_out)
        print(f"WROTE {mem_out} ({mem.height} rows)")

        mpdf = mem.with_columns(peak_gib=pl.col("peak_rss_bytes") / 2**30).to_pandas()
        g = sns.relplot(
            data=mpdf, x="batch_size", y="peak_gib", hue="dataset_norm", style="dl_mode",
            col="seqlen", col_wrap=2, kind="line", marker="o",
            facet_kws={"sharex": False, "sharey": False},
        )
        for ax in g.axes.flat:
            ax.set_xscale("log", base=2)
        g.set_axis_labels("batch_size", "peak RSS (GiB)")
        g.savefig(fig_dir / "gvl027_peak_rss.png", dpi=150, bbox_inches="tight")
        print(f"WROTE {fig_dir / 'gvl027_peak_rss.png'}")
    else:
        print("No memory CSVs found under haps_memory/ or tracks_memory/ — skipping memory report.")


if __name__ == "__main__":
    cyclopts.run(main)
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `cd hap_track_throughput/bin_gvl027 && pixi run -e bench027 python -m pytest tests/test_compare_helpers.py -v`
Expected: all PASS.

- [ ] **Step 5: Verify the existing probe-common tests still pass (no regression)**

Run: `cd hap_track_throughput/bin_gvl027 && pixi run -e bench027 python -m pytest -v`
Expected: `test_probe_common.py` + `test_compare_helpers.py` all PASS.

- [ ] **Step 6: Commit**

```bash
git add hap_track_throughput/bin_gvl027/compare_to_baseline.py hap_track_throughput/bin_gvl027/tests/test_compare_helpers.py
git commit -m "feat(compare): read nf full-bench layout, normalize dataset, memory absolute report"
```

---

## Task 10: Smoke test the pipeline end-to-end

**Files:**
- None created/modified — this is a verification task run from `hap_track_throughput/`.

Before any full launch, run the tiny `--test_grid` smoke for one dataset across both `dl_mode`s. This confirms: the `gvl027` env activates on the compute node, SVAR conversion applies the filters, `gvl.write(variants=<svar>)` succeeds, and the haps script emits the reconciled schema with finite throughput on the small cells. The 1kgp dataset is the cheapest haps-only smoke (no bigwig).

- [ ] **Step 1: Launch the smoke run (throughput, haps only, tiny grid)**

Run:
```bash
cd hap_track_throughput
nextflow run benchmark.nf -profile gvl027 -c configs/1kgp.config \
  --test_grid --bench_haps --results_dir "$PWD/../results_gvl027_smoke" \
  --ds_dir "$PWD/data/datasets_smoke/1kgp"
```
Expected: pipeline completes; `BENCH_SVAR_CONVERT`, `BENCH_WRITE_DATASET` (4 lengths × svar), and `BENCH_HAPS` (4 lengths × 2 dl_modes = 8 tasks) all succeed.

- [ ] **Step 2: Confirm the output layout and schema**

Run:
```bash
ls ../results_gvl027_smoke/haps/
head -1 ../results_gvl027_smoke/haps/1KGP_2048_svar_none.csv
```
Expected: files named `1KGP_{2048,16384,131072,1048576}_svar_{none,buffered}.csv` exist; the header is exactly:
`dataset,backend,dl_mode,threads,seqlen,batch_size,n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)`

- [ ] **Step 3: Confirm no spurious NaN on the small cells**

Run:
```bash
pixi run -e bench027 python - <<'PY'
import polars as pl, glob
for p in sorted(glob.glob("../results_gvl027_smoke/haps/*_none.csv")):
    df = pl.read_csv(p)
    t = df["throughput (MiB/s)"].cast(pl.Float64, strict=False)
    print(p, "rows", df.height, "finite", int(t.is_finite().sum()))
PY
```
Expected: every `_none.csv` has all rows finite (NaN only legitimately appears for buffered cells where a single minibatch exceeds `buffer_bytes`).

- [ ] **Step 4: Smoke the memory pass (one length, tiny grid)**

Run:
```bash
cd hap_track_throughput
nextflow run benchmark.nf -profile gvl027 -c configs/1kgp.config \
  --test_grid --bench_haps --measure_memory \
  --results_dir "$PWD/../results_gvl027_smoke" \
  --ds_dir "$PWD/data/datasets_smoke/1kgp" -resume
```
Expected: completes; `../results_gvl027_smoke/haps_memory/1KGP_*_svar_*.csv` exist with header
`dataset,backend,dl_mode,threads,seqlen,batch_size,avg_rss_bytes,peak_rss_bytes` and non-NaN `peak_rss_bytes`.

- [ ] **Step 5: Smoke the comparison script against the smoke outputs**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper
pixi run -e bench027 python hap_track_throughput/bin_gvl027/compare_to_baseline.py \
  --results-dir results_gvl027_smoke --fig-dir figures_smoke
```
Expected: prints `WROTE results_gvl027_smoke/parity_summary.csv`, `buffered_throughput.csv`, `memory_summary.csv`, and the two PNGs; the median throughput-ratio table prints for `(1kgp, haps)`. No traceback.

- [ ] **Step 6: Clean up the smoke artifacts**

Run: `rm -rf results_gvl027_smoke figures_smoke hap_track_throughput/data/datasets_smoke hap_track_throughput/work hap_track_throughput/.nextflow`
Expected: smoke outputs removed (real runs use `results_gvl027/` and the per-config `ds_dir`). No commit — these are throwaway.

> If any smoke step fails, STOP and use superpowers:systematic-debugging before proceeding to the full launch. The most likely failure points (per the design's risk list) are env activation in `beforeScript` (Task 8) and the buffered empty-epoch guard on small grids (Task 2).

---

## Task 11: Full-run driver + docs

**Files:**
- Create: `hap_track_throughput/run_full_bench.sbatch`
- Modify: `CLAUDE.md`

A single sbatch driver that runs both pipeline invocations (throughput, then memory) for each dataset config, then the comparison. The pipeline's own SLURM executor fans the grid out; this driver is just the launcher and need only request a tiny head-node allocation (Nextflow stays resident submitting/monitoring child jobs).

- [ ] **Step 1: Write the driver**

Create `hap_track_throughput/run_full_bench.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=gvl027-full
#SBATCH --partition=carter-compute
#SBATCH --account=carter
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl027/full_%j.log
#SBATCH --error=/carter/users/dlaub/projects/gvl-paper/results_gvl027/full_%j.log
#
# Full GVL 0.27.0 throughput + memory bench across 1kgp, tcga-atac, ukbb.
# This is the Nextflow HEAD job: it stays resident and submits the grid as child
# SLURM jobs (executor=slurm via -profile gvl027). Each pipeline invocation is
# idempotent under -resume; re-submitting resumes from cached work.
#   throughput pass: default to_dataloader + single buffered loader
#   memory pass:     --measure_memory (peak/avg RSS), via -resume reusing SVAR+write
set -euo pipefail

export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT/hap_track_throughput"

# bench_tracks is enabled per-config (tcga-atac.config sets it true); haps always on.
CONFIGS=(configs/1kgp.config configs/tcga-atac.config configs/ukbb.config)

for cfg in "${CONFIGS[@]}"; do
  echo "=== THROUGHPUT $cfg  $(date) ==="
  nextflow run benchmark.nf -profile gvl027 -c "$cfg" -resume

  echo "=== MEMORY $cfg  $(date) ==="
  nextflow run benchmark.nf -profile gvl027 -c "$cfg" --measure_memory -resume
done

echo "=== COMPARE  $(date) ==="
cd "$ROOT"
pixi run -e bench027 python hap_track_throughput/bin_gvl027/compare_to_baseline.py \
  || echo "WARN: compare failed — rerun manually; result CSVs are intact"

echo "=== DONE  $(date) ==="
```

- [ ] **Step 2: Shellcheck / dry-validate the driver (no submission)**

Run: `bash -n hap_track_throughput/run_full_bench.sbatch && echo "syntax OK"`
Expected: `syntax OK`. (Do not `sbatch` it as part of plan execution — the user submits the real multi-day run.)

- [ ] **Step 3: Document the harness in CLAUDE.md**

In `CLAUDE.md`, under the `feature.bench027` bullet in the "GenVarLoader version sensitivity" section, append this paragraph after the existing `**Why 0.27.0:**` note:

```markdown
  **Full throughput+memory bench (2026-06-05):** the same `bench027` env now also
  backs the *full* manuscript grid (not just the reduced probe), run through the
  production Nextflow harness `hap_track_throughput/benchmark.nf` with
  `-profile gvl027` (see `hap_track_throughput/nextflow.config`, which activates
  this env on each SLURM job). All GVL datasets are **SVAR-backed** (`bench_native`
  off by default); genoray 2.9.0's hap-safe filter (`~is_symbolic & ~is_breakend`,
  via `bin/_genoray_filter.py`) is applied at SVAR conversion. Each cell is run in
  two dataloader modes (`none` = default `to_dataloader`, `buffered` =
  `buffer_bytes=2 GiB`). `bin/benchmark_haps.py` / `benchmark_tracks.py` emit the
  reconciled schema `dataset,backend,dl_mode,threads,seqlen,batch_size,
  n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)` (memory pass:
  `...,avg_rss_bytes,peak_rss_bytes`), directly comparable to the 0.6.1 baseline.
  Drive the whole thing with `hap_track_throughput/run_full_bench.sbatch`; outputs
  land in `results_gvl027/{haps,tracks,haps_memory,tracks_memory,...}` and are joined
  to the baseline by `bin_gvl027/compare_to_baseline.py`. The throughput-vs-0.6.1
  ratio is an **internal sanity check**; memory is reported as 0.27.0 absolute (no
  0.6.1 memory baseline exists). Do **not** repoint the manuscript baseline CSVs to
  this env.
```

- [ ] **Step 4: Commit**

```bash
git add hap_track_throughput/run_full_bench.sbatch CLAUDE.md
git commit -m "feat(bench): full-bench sbatch driver + document gvl027 full harness"
```

---

## Self-Review

**Spec coverage** (each spec section → task):

- SVAR-only writes / `bench_native` gate → Task 7 Step 2. ✓
- `dl_mode ∈ {none, buffered}` cross + thread `--dl-mode` + `_${dl_mode}` filenames → Task 7 Steps 2,3,5,6,7. ✓
- genoray filter params (`no_symbolic`, `no_breakend`) into `BENCH_SVAR_CONVERT` → Task 7 Step 4 + Task 5. ✓
- `results_dir` → `results_gvl027/` → Task 7 Step 1. ✓
- `gvl027` Nextflow profile (env activation, Nextflow not in pixi) → Task 8. ✓
- genoray 2.9.0 filter at conversion (PGEN `filter=`, VCF `filter=`+`pl_filter=`) → Task 5. ✓
- `bench_svar_vcf_plink.py` correctness-only fix → Task 6. ✓
- Throughput schema reconciliation (total_bytes, duration_ns, MiB/s; lift probe helpers) → Tasks 2,3,4. ✓
- Memory schema gains `dl_mode` → Tasks 3,4 (`MEMORY_HEADER`). ✓
- buffered haps `deterministic=True`; buffered `ValueError`→NaN; empty-epoch guard → Tasks 2 (`measure_cell`), 3, 4. ✓
- Outputs layout + `compare_to_baseline.py` (mode from dir, dl_mode from filename/column; internal throughput join; buffered standalone; memory absolute) → Task 9. ✓
- Pin `genoray ==2.9.0` in bench027 + bench/default; verify lock → Task 1. ✓
- Smoke test (`-profile gvl027 --test_grid`, both dl_modes, filter applied, schema, no NaN) → Task 10. ✓
- `bench_svar_vcf_plink.py` import smoke → Task 6 Step 3. ✓
- Full run cost / carter-cn-04 pin / cpus=64 → already in `benchmark.nf` `BENCH_HAPS`/`BENCH_TRACKS` directives (`clusterOptions '--nodelist=carter-cn-04'`, `cpus 64`); driver in Task 11. ✓

**Placeholder scan:** No TBD/TODO/"add error handling"/"similar to" — every code step is complete. ✓

**Type/name consistency:**
- `measure_cell` / `CellResult` / `n_bytes` / `mib_per_s` / `THROUGHPUT_HEADER` / `MEMORY_HEADER` — defined in Task 2, used identically in Tasks 3,4. ✓
- `hap_safe_pl_filter` / `hap_safe_vcf_callable` / `open_filtered_reader` — defined in Task 5, used in Tasks 5,6. ✓
- `norm_dataset` / `parse_mode_dir` — defined and tested in Task 9. ✓
- nf record fields `dl_mode` added to `Dataset` and `BenchResult`; producer (`dl_inputs.map`) and consumers (`BENCH_HAPS`/`BENCH_TRACKS` output records, output{} path closures) all reference `r.dl_mode` / `ds.dl_mode` consistently. ✓
- CLI flag names: scripts use `--dl-mode`, `--no-symbolic`, `--no-breakend` (cyclopts kebab-cases `dl_mode`/`no_symbolic`/`no_breakend`); nf passes those exact strings. ✓
