# Variant-throughput targeted steady-state (speed fix) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the variant-throughput benchmark's speed by applying the sustained steady-state loop only to SVAR (where the numba parking artifact lives) and reverting BCF/PGEN/PRESUB-BCF to a single timed pass per replicate, plus shrinking the per-replicate stream from 64 to 8 batches.

**Architecture:** `bench_svar.py` and `_streaming.py` are untouched — SVAR keeps its AOT search + sustained `_svar_pack` loop. The three I/O bench scripts drop `run_stream`/`prime`/`drive_loop` and the `min_seconds`/`min_batches` params, timing one pass over the replicate's batches instead (throughput) or one pass under `PeakRssSampler` (memory). `variant_throughput.nf` removes the two `--min-*` flags from the six I/O `BENCH_*` processes and lowers `stream_batches` default to 8. CSV schema and plotting are unchanged.

**Tech Stack:** Python 3.12, polars, numpy, cyvcf2, genoray (`VCF`/`PGEN`), cyclopts; orchestrated by typed Nextflow (`variant_throughput.nf`); run under pixi (default env).

**Spec:** `docs/superpowers/specs/2026-06-09-variant-throughput-targeted-steadystate-design.md`

**Conventions for every command below:** run from repo root `/carter/users/dlaub/projects/gvl-paper`; prefix Python with `pixi run`. Nextflow `bin/` is on PATH at runtime, so the bench scripts import sibling modules (`_pairs`, `_mem_sampler`) by bare name.

**Note on testing:** the three I/O bench scripts read real genomic files and have no standalone unit tests (only `_streaming.py` and `_pairs.py` are unit-tested, and both are untouched here). Per-script verification is therefore a `--help` parse check confirming the `--min-*` flags are gone; end-to-end correctness is the smoke run in Task 5.

---

## File structure

- Modify `variant_throughput/bin/bench_bcf.py` — single timed pass; drop streaming imports + `min_*` params.
- Modify `variant_throughput/bin/bench_pgen.py` — same.
- Modify `variant_throughput/bin/bench_presubset_bcf.py` — keep AOT subset (→ `setup_ns`) + cleanup; single timed read pass; drop streaming imports + `min_*` params.
- Modify `variant_throughput/variant_throughput.nf` — `stream_batches` default 64→8; remove `--min-seconds`/`--min-batches` from the six I/O `BENCH_*` script blocks (keep them on the two SVAR blocks).

`_streaming.py`, `bench_svar.py`, `generate_pairs.py`, `_pairs.py`, the tests, and all plotting/config files are **unchanged**.

---

## Task 1: `bench_bcf.py` — single timed pass

**Files:**
- Modify: `variant_throughput/bin/bench_bcf.py`

- [ ] **Step 1: Replace the whole file**

Overwrite `variant_throughput/bin/bench_bcf.py` with:

```python
#! /usr/bin/env python

from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from cyclopts import run

from _pairs import split_pair_batches


def bench(
    pairs_parquet: Path,
    bcf: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
):
    import polars as pl
    from genoray import VCF

    df = pl.read_parquet(pairs_parquet)
    q_len = int(df["end"][0] - df["start"][0])

    rows_out: list[dict] = []

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
            t0 = perf_counter_ns()
            n_calls = sum(gather(pairs) for pairs in batches)
            elapsed_ns = perf_counter_ns() - t0
            rows_out.append({
                "dataset": dataset or bcf.name,
                "method": "bcf",
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
                "dataset": dataset or bcf.name,
                "method": "bcf",
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
```

- [ ] **Step 2: Verify it parses and the `--min-*` flags are gone**

Run: `pixi run python variant_throughput/bin/bench_bcf.py --help`
Expected: help text with `--mode`, `--n-samples`; **no** `--min-seconds` or `--min-batches`; no import error.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/bench_bcf.py
git commit -m "fix(variant-throughput): BCF single timed pass (drop sustained loop)"
```

---

## Task 2: `bench_pgen.py` — single timed pass

**Files:**
- Modify: `variant_throughput/bin/bench_pgen.py`

- [ ] **Step 1: Replace the whole file**

Overwrite `variant_throughput/bin/bench_pgen.py` with:

```python
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
```

- [ ] **Step 2: Verify it parses and the `--min-*` flags are gone**

Run: `pixi run python variant_throughput/bin/bench_pgen.py --help`
Expected: help text with `--mode`, `--n-samples`; **no** `--min-seconds` or `--min-batches`; no import error.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/bench_pgen.py
git commit -m "fix(variant-throughput): PGEN single timed pass (drop sustained loop)"
```

---

## Task 3: `bench_presubset_bcf.py` — AOT subset + single timed read pass

**Files:**
- Modify: `variant_throughput/bin/bench_presubset_bcf.py`

Keep `_subset_pairs`, `_read_subsets`, the AOT subset timing (`setup_ns`), and the
temp-BCF cleanup. Only the timed-replay portion changes from a sustained loop to a
single pass.

- [ ] **Step 1: Replace the whole file**

Overwrite `variant_throughput/bin/bench_presubset_bcf.py` with:

```python
#! /usr/bin/env python

import os
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from cyclopts import run

from _pairs import split_pair_batches


def _subset_pairs(
    bcf: Path,
    pairs: list[tuple[tuple[str, int, int], str]],
    tmp_dir: Path,
) -> list[str]:
    """Run bcftools view to pre-subset each pair into a temp BCF. Returns tmp paths."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths: list[str] = []
    for (contig, start, end), sample in pairs:
        fd, path = tempfile.mkstemp(suffix=".bcf", dir=tmp_dir)
        os.close(fd)
        tmp_paths.append(path)
        subprocess.run(
            [
                "bcftools",
                "view",
                "-s",
                sample,
                "-r",
                f"chr{contig}:{start + 1}-{end}",
                "--min-ac",
                "1",
                "--no-update",
                "-Ob",
                "-o",
                path,
                str(bcf),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return tmp_paths


def _read_subsets(tmp_paths: list[str]) -> int:
    """Read pre-subsetted BCFs with cyvcf2. Returns n_calls."""
    import numpy as np
    import cyvcf2

    n_calls = 0
    for path in tmp_paths:
        vcf = cyvcf2.VCF(path)
        chunks = []
        for v in vcf:
            chunks.append(v.genotype.array())
        if chunks:
            n_calls += int((np.concatenate(chunks, axis=0)[:, :2] > 0).sum())
        vcf.close()
    return n_calls


def bench(
    pairs_parquet: Path,
    bcf: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
):
    import polars as pl

    df = pl.read_parquet(pairs_parquet)
    q_len = int(df["end"][0] - df["start"][0])
    tmp_dir = Path(".bench_tmp")

    rows_out: list[dict] = []

    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        batches = split_pair_batches(group)
        if not batches:
            continue
        n_pairs = sum(len(pairs) for pairs in batches)

        # AOT: pre-subset each batch into temp BCFs (not re-done per epoch); timed once -> setup_ns.
        batch_tmp: list[list[str]] = []

        def gather(tmp_paths) -> int:
            return _read_subsets(tmp_paths)

        try:
            t0 = perf_counter_ns()
            for pairs in batches:
                batch_tmp.append(_subset_pairs(bcf, pairs, tmp_dir))
            setup_ns = perf_counter_ns() - t0

            if mode == "throughput":
                t0 = perf_counter_ns()
                n_calls = sum(gather(tp) for tp in batch_tmp)
                elapsed_ns = perf_counter_ns() - t0
                rows_out.append({
                    "dataset": dataset or bcf.name,
                    "method": "presubset_bcf",
                    "query_length": q_len,
                    "n_samples": int(n_samples),
                    "replicate": int(rep),
                    "n_pairs": n_pairs,
                    "n_calls": n_calls,
                    "elapsed_ns": elapsed_ns,
                    "setup_ns": setup_ns,
                })
            else:
                from _mem_sampler import PeakRssSampler

                with PeakRssSampler() as s:
                    n_calls = sum(gather(tp) for tp in batch_tmp)
                rows_out.append({
                    "dataset": dataset or bcf.name,
                    "method": "presubset_bcf",
                    "query_length": q_len,
                    "n_samples": int(n_samples),
                    "replicate": int(rep),
                    "n_pairs": n_pairs,
                    "n_calls": n_calls,
                    "peak_rss_bytes": s.peak,
                })
        finally:
            for tmp_paths in batch_tmp:
                for path in tmp_paths:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass

    pl.DataFrame(rows_out).write_csv(output)


if __name__ == "__main__":
    run(bench)
```

- [ ] **Step 2: Verify it parses and the `--min-*` flags are gone**

Run: `pixi run python variant_throughput/bin/bench_presubset_bcf.py --help`
Expected: help text with `--mode`, `--n-samples`; **no** `--min-seconds` or `--min-batches`; no import error.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/bench_presubset_bcf.py
git commit -m "fix(variant-throughput): PRESUB-BCF single timed read pass (drop sustained loop)"
```

---

## Task 4: Nextflow — lower `stream_batches`, strip `--min-*` from I/O processes

**Files:**
- Modify: `variant_throughput/variant_throughput.nf`

The two SVAR processes (`BENCH_SVAR_THROUGHPUT`, `BENCH_SVAR_MEMORY`) **keep** their
`--min-seconds`/`--min-batches` flags. Only the six I/O processes lose them.

- [ ] **Step 1: Lower the `stream_batches` default**

In the top-level `params { ... }` block (line 14), change:

```groovy
stream_batches: Integer = 64
```

to:

```groovy
stream_batches: Integer = 8
```

- [ ] **Step 2: Strip `--min-*` from `BENCH_BCF_THROUGHPUT`**

Replace:

```groovy
      bcf_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
```

with:

```groovy
      bcf_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples}
```

- [ ] **Step 3: Strip `--min-*` from `BENCH_BCF_MEMORY`**

Replace:

```groovy
      bcf_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
```

with:

```groovy
      bcf_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples}
```

- [ ] **Step 4: Strip `--min-*` from `BENCH_PGEN_THROUGHPUT`**

Replace:

```groovy
      pgen_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
```

with:

```groovy
      pgen_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples}
```

- [ ] **Step 5: Strip `--min-*` from `BENCH_PGEN_MEMORY`**

Replace:

```groovy
      pgen_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
```

with:

```groovy
      pgen_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples}
```

- [ ] **Step 6: Strip `--min-*` from `BENCH_PRESUBSET_BCF_THROUGHPUT`**

Replace:

```groovy
      presubset_bcf_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
```

with:

```groovy
      presubset_bcf_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples}
```

- [ ] **Step 7: Strip `--min-*` from `BENCH_PRESUBSET_BCF_MEMORY`**

Replace:

```groovy
      presubset_bcf_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
```

with:

```groovy
      presubset_bcf_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples}
```

- [ ] **Step 8: Confirm the SVAR processes still carry `--min-*`**

Run: `grep -c "min-seconds" variant_throughput/variant_throughput.nf`
Expected: `2` (only `BENCH_SVAR_THROUGHPUT` and `BENCH_SVAR_MEMORY`).

- [ ] **Step 9: Verify the Nextflow config/script parses**

Run: `cd variant_throughput && pixi run nextflow config -c configs/1kgp.config >/dev/null && echo PARSE_OK; cd ..`
Expected: `PARSE_OK` (no DSL2 syntax/param error).

- [ ] **Step 10: Commit**

```bash
git add variant_throughput/variant_throughput.nf
git commit -m "fix(variant-throughput): stream_batches 64->8; drop min-* flags from I/O benches"
```

---

## Task 5: End-to-end smoke run + verification

**Files:** none modified — validates the integrated pipeline against small data.

**Precondition:** the smoke overlay (`configs/smoke.config`) restricts to two short
query lengths and tiny cohorts. `stream_batches=4` is already set there (overrides
the new default of 8), which is correct for a fast smoke run.

- [ ] **Step 1: Run the smoke pipeline**

Run:
```bash
cd variant_throughput && pixi run nextflow run variant_throughput.nf -c configs/1kgp.config -c configs/smoke.config -resume; cd ..
```
Expected: all `GENERATE_PAIRS*`, `BENCH_*`, `COMBINE_*`, and `PLOT_*` processes complete; `results/` gains the throughput + memory CSVs and plots. No process hangs.

- [ ] **Step 2: Verify the throughput CSV schema is unchanged**

Run:
```bash
pixi run python -c "import polars as pl; df=pl.read_csv('variant_throughput/results/svar_throughput.csv'); print(df.columns)"
```
Expected columns exactly: `dataset, method, query_length, n_samples, replicate, n_pairs, n_calls, elapsed_ns, setup_ns`.

- [ ] **Step 3: Verify I/O-format rows have positive timings and `setup_ns` only where expected**

Run:
```bash
pixi run python -c "
import polars as pl
for m in ['bcf','pgen']:
    df = pl.read_csv(f'variant_throughput/results/{m}_throughput.csv')
    assert (df['n_calls'] > 0).all(), f'{m}: non-positive n_calls'
    assert (df['elapsed_ns'] > 0).all(), f'{m}: non-positive elapsed_ns'
    assert df['setup_ns'].is_null().all(), f'{m}: setup_ns should be null'
    print(f'OK {m}: positive n_calls/elapsed_ns, null setup_ns')
ps = pl.read_csv('variant_throughput/results/presubset_bcf_throughput.csv')
assert (ps['n_calls'] > 0).all() and (ps['elapsed_ns'] > 0).all()
assert (ps['setup_ns'] > 0).all(), 'presubset: setup_ns should be positive'
print('OK presubset_bcf: positive n_calls/elapsed_ns/setup_ns')
"
```
Expected: `OK bcf ...`, `OK pgen ...`, `OK presubset_bcf ...`.

- [ ] **Step 4: Verify the SVAR steady-state fix is still intact**

Run:
```bash
pixi run python -c "
import polars as pl
df = pl.read_csv('variant_throughput/results/svar_throughput.csv')
assert (df['elapsed_ns'] > 0).all() and (df['setup_ns'] > 0).all()
g = (df.group_by('query_length','n_samples')
       .agg((pl.col('elapsed_ns').max()/pl.col('elapsed_ns').min()).alias('spread')))
print(g.sort('spread', descending=True).head())
assert g['spread'].max() < 10, 'SVAR elapsed_ns bimodal within a cell'
print('OK: SVAR elapsed_ns unimodal within cells')
"
```
Expected: `OK: SVAR elapsed_ns unimodal within cells`.

- [ ] **Step 5: Run the unit-test suite (unchanged modules still pass)**

Run: `pixi run pytest variant_throughput/bin/tests/ -v`
Expected: all tests in `test_streaming.py` and `test_pairs.py` PASS.

- [ ] **Step 6: Restore smoke-overwritten results**

The smoke run overwrites `results/*` with tiny smoke data — do NOT keep it as the
manuscript figures. If `results/` is tracked and dirty:
```bash
git checkout -- variant_throughput/results
```
The real regeneration (full `configs/1kgp.config` without the smoke overlay) is a
separate operational step, not part of this plan.

---

## Self-review notes

- **Spec coverage:** SVAR unchanged (stated in plan intro + Task 4 keeps its flags);
  BCF/PGEN single pass (Tasks 1–2); PRESUB-BCF AOT + single read pass (Task 3);
  `stream_batches` 64→8 (Task 4 Step 1); `--min-*` removed from six I/O processes
  (Task 4 Steps 2–7); CSV schema + plotting unchanged (verified Task 5 Steps 2–3);
  smoke run completes without hangs (Task 5 Step 1). All spec design sections map to
  a task.
- **Out of scope** (per spec): regenerating committed manuscript results (Task 5
  Step 6 restores `results/`), grid changes, `_streaming.py`/`bench_svar.py`
  internals (untouched).
- **Type/name consistency:** the three I/O scripts keep the same row dict keys as
  before (`dataset, method, query_length, n_samples, replicate, n_pairs, n_calls,
  elapsed_ns, setup_ns` for throughput; `peak_rss_bytes` instead of
  `elapsed_ns`/`setup_ns` for memory), so the unchanged `COMBINE_*`/`PLOT_*` and the
  CSV schema stay valid. `bench()` signatures drop `min_seconds`/`min_batches`
  consistently with the nf flag removal in Task 4.
```
