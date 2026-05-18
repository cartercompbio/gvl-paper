# Variant Throughput Sample-Size Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Required reading before any `.nf` edit:** the `writing-typed-nextflow` skill. This pipeline runs on typed Nextflow 26.04+; do not regress to legacy DSL2 patterns (`tuple val`, `Channel.from`, implicit `it`, `set`/`tap`, `publishDir`, multi-arg `mix`, `groupBy { closure }`, `(name): Tuple<...>` input destructuring).

**Goal:** Add an orthogonal sample-size (N) sweep to `variant_throughput/variant_throughput.nf` so each of the four methods (`svar`, `bcf`, `pgen`, `presubset_bcf`) is benchmarked for throughput and peak-RSS as a function of cohort size, at a fixed query length, in addition to the existing query-length sweep.

**Architecture:** Two parallel sub-workflow paths feed the same `COMBINE_*` / `PLOT_*` stages. The N path builds N-sized subset files once per N (untimed) via `bcftools view -S`, `plink2 --keep`, and `SparseVar.from_pgen`, then runs the unmodified bench scripts against those files. Each bench CSV row carries an `n_samples` column so the combined output is filterable into either sweep.

**Tech Stack:** Nextflow 26.04+ typed strict syntax, Python 3.12 (pixi `bench` env), `genoray` `SparseVar`/`VCF`/`PGEN`, `polars`, `bcftools`, `plink2`, `seaborn`.

**Spec:** `docs/superpowers/specs/2026-05-18-variant-throughput-sample-size-sweep-design.md`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `variant_throughput/bin/bench_svar.py` | Modify | Accept `--n-samples` and write into every CSV row. |
| `variant_throughput/bin/bench_bcf.py` | Modify | Same. |
| `variant_throughput/bin/bench_pgen.py` | Modify | Same. |
| `variant_throughput/bin/bench_presubset_bcf.py` | Modify | Same. |
| `variant_throughput/bin/make_sample_list.py` | Create | Deterministic shuffle of `_svar.available_samples`, emit first N to a text file (one per line). Also has `--print-total` mode to report the full cohort count. |
| `variant_throughput/bin/plot_throughput.py` | Modify | Detect both sweeps in the combined CSV; emit query-length-axis and N-axis figures. |
| `variant_throughput/bin/plot_memory.py` | Modify | Same. |
| `variant_throughput/variant_throughput.nf` | Modify | New params, new records, new processes, new outputs. Refactor existing bench processes to take a `SweepInput` record so the same processes serve both the q-len and N paths. |
| `variant_throughput/configs/1kgp.config` | No change | Defaults live in the workflow. |

Pipeline-level decomposition:

- **Per-N untimed setup chain:** `MAKE_SAMPLE_LIST` → `SUBSET_BCF` + `SUBSET_PGEN` → `BUILD_SVAR_FROM_PGEN` → `GENERATE_PAIRS_N`.
- **Per-q-len existing chain (unchanged behavior):** `GENERATE_PAIRS` → 8 bench processes.
- Both chains terminate in `SweepInput` records that feed the same 8 bench processes (refactored to read paths from the record, not `params`).
- Combined CSVs and plots are produced once each, covering both sweeps.

---

## Task 1: Add `--n-samples` to `bench_svar.py`

**Files:**
- Modify: `variant_throughput/bin/bench_svar.py`

- [ ] **Step 1: Add the parameter and propagate it into output rows**

Edit `variant_throughput/bin/bench_svar.py`. Add `n_samples: int = 0` to the `bench` signature and include it in every row dict.

```python
def bench(
    pairs_parquet: Path,
    svar: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    use_custom_pack: bool = True,
    n_samples: int = 0,
):
```

In both `rows_out.append({...})` calls inside the throughput and memory branches, insert `"n_samples": int(n_samples),` immediately after the existing `"query_length": q_len,` line.

- [ ] **Step 2: Smoke-check the script parses**

Run: `pixi r python variant_throughput/bin/bench_svar.py --help`

Expected: cyclopts help text that lists `--n-samples` as an option, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/bench_svar.py
git commit -m "feat(variant_throughput): bench_svar accepts --n-samples"
```

---

## Task 2: Add `--n-samples` to `bench_bcf.py`

**Files:**
- Modify: `variant_throughput/bin/bench_bcf.py`

- [ ] **Step 1: Add the parameter and propagate**

Edit `variant_throughput/bin/bench_bcf.py`. Update the signature:

```python
def bench(
    pairs_parquet: Path,
    bcf: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
):
```

Insert `"n_samples": int(n_samples),` directly after `"query_length": q_len,` in both `rows_out.append` blocks.

- [ ] **Step 2: Smoke-check**

Run: `pixi r python variant_throughput/bin/bench_bcf.py --help`

Expected: help shows `--n-samples`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/bench_bcf.py
git commit -m "feat(variant_throughput): bench_bcf accepts --n-samples"
```

---

## Task 3: Add `--n-samples` to `bench_pgen.py`

**Files:**
- Modify: `variant_throughput/bin/bench_pgen.py`

- [ ] **Step 1: Add the parameter and propagate**

Edit `variant_throughput/bin/bench_pgen.py`. Update the signature:

```python
def bench(
    pairs_parquet: Path,
    pgen: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
):
```

Insert `"n_samples": int(n_samples),` directly after `"query_length": q_len,` in both `rows_out.append` blocks.

- [ ] **Step 2: Smoke-check**

Run: `pixi r python variant_throughput/bin/bench_pgen.py --help`

Expected: help shows `--n-samples`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/bench_pgen.py
git commit -m "feat(variant_throughput): bench_pgen accepts --n-samples"
```

---

## Task 4: Add `--n-samples` to `bench_presubset_bcf.py`

**Files:**
- Modify: `variant_throughput/bin/bench_presubset_bcf.py`

- [ ] **Step 1: Add the parameter and propagate**

Edit `variant_throughput/bin/bench_presubset_bcf.py`. Update the signature:

```python
def bench(
    pairs_parquet: Path,
    bcf: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    n_samples: int = 0,
):
```

Insert `"n_samples": int(n_samples),` directly after `"query_length": q_len,` in both `rows_out.append` blocks.

- [ ] **Step 2: Smoke-check**

Run: `pixi r python variant_throughput/bin/bench_presubset_bcf.py --help`

Expected: help shows `--n-samples`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/bench_presubset_bcf.py
git commit -m "feat(variant_throughput): bench_presubset_bcf accepts --n-samples"
```

---

## Task 5: Create `make_sample_list.py`

**Files:**
- Create: `variant_throughput/bin/make_sample_list.py`

- [ ] **Step 1: Write the script**

Create `variant_throughput/bin/make_sample_list.py`:

```python
#! /usr/bin/env python

import random
import sys
from pathlib import Path

from cyclopts import run


def main(
    svar: Path,
    output: Path,
    n: int = 0,
    seed: int = 0,
    print_total: bool = False,
):
    """Emit a deterministic subset of samples from a SparseVar dataset.

    With --print-total, prints the total sample count to stdout and exits
    without writing the output file.

    Otherwise: shuffles _svar.available_samples with `seed`, takes the first
    `n` IDs, and writes them one-per-line to `output`. If `n` is 0 or >= total,
    writes every sample (full cohort passthrough).
    """
    from genoray import SparseVar

    _svar = SparseVar(svar)
    available = list(_svar.available_samples)

    if print_total:
        sys.stdout.write(f"{len(available)}\n")
        return

    if n <= 0 or n >= len(available):
        chosen = available
    else:
        rng = random.Random(seed)
        shuffled = available.copy()
        rng.shuffle(shuffled)
        chosen = shuffled[:n]

    output.write_text("\n".join(chosen) + "\n")


if __name__ == "__main__":
    run(main)
```

- [ ] **Step 2: Make it executable and smoke-check**

```bash
chmod +x variant_throughput/bin/make_sample_list.py
pixi r python variant_throughput/bin/make_sample_list.py --help
```

Expected: cyclopts help text exits 0, lists `--n`, `--seed`, `--print-total`.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/make_sample_list.py
git commit -m "feat(variant_throughput): add make_sample_list.py for N sweep"
```

---

## Task 6: Extend `plot_throughput.py` with an N-axis figure

**Files:**
- Modify: `variant_throughput/bin/plot_throughput.py`

The combined CSV will contain rows from both sweeps. Each sweep is identifiable by which axis it varies:
- **q-len sweep rows:** `n_samples` is constant (full cohort), `query_length` varies.
- **N sweep rows:** `query_length` is constant (`params.n_sweep_query_length`), `n_samples` varies.

Strategy: for each axis, slice the combined CSV to rows where that axis varies (i.e. has more than one distinct value within the rest of the grouping), emit the existing plot for the q-len axis and a parallel plot keyed on `log10_n_samples` for the N axis.

- [ ] **Step 1: Refactor the plotting into two passes (q-len axis and N axis)**

Replace the body of `bench(...)` in `variant_throughput/bin/plot_throughput.py` with:

```python
def bench(*csvs: Path, output_dir: Path = Path("results")):
    import numpy as np
    import polars as pl
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.5)

    if not csvs:
        csvs = tuple(output_dir / f"{m}_throughput.csv" for m in METHOD_LABELS)

    schema_overrides = {"setup_ns": pl.Int64}
    raw = pl.concat([pl.read_csv(p, schema_overrides=schema_overrides) for p in csvs])

    # ---- q-len axis (rows where n_samples is at its max, i.e. full cohort) ----
    full_n = raw["n_samples"].max()
    qlen_df = (
        raw.filter(pl.col("n_samples") == full_n)
        .filter((pl.col("n_calls") > 0) & (pl.col("elapsed_ns") > 0))
        .with_columns(
            (pl.col("n_calls") / (pl.col("elapsed_ns") * 1e-9)).alias("calls_per_sec"),
        )
        .with_columns(
            pl.col("query_length").log(base=10).alias("log10_query_length"),
            pl.col("calls_per_sec").log(base=10).alias("log10_calls_per_sec"),
            pl.col("method").replace(METHOD_LABELS).alias("method_label"),
        )
        .to_pandas()
    )

    if not qlen_df.empty:
        hue_order = (
            qlen_df.groupby("method_label")["calls_per_sec"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        lmplot(
            qlen_df,
            x_col="log10_query_length",
            y_col="log10_calls_per_sec",
            x_label="Query length (bp)",
            y_label="Throughput (alt calls / sec)",
            hue_order=hue_order,
            output_dir=output_dir,
            stem="plot",
        )

    # ---- N axis (rows where query_length is at its max in the N sweep — i.e. the
    #      fixed N-sweep query length; we identify it as the query_length value
    #      whose rows contain more than one distinct n_samples value) ----
    n_distinct_by_q = raw.group_by("query_length").agg(
        pl.col("n_samples").n_unique().alias("n_unique_samples")
    )
    n_sweep_qlens = n_distinct_by_q.filter(pl.col("n_unique_samples") > 1)["query_length"].to_list()

    if n_sweep_qlens:
        n_df = (
            raw.filter(pl.col("query_length").is_in(n_sweep_qlens))
            .filter((pl.col("n_calls") > 0) & (pl.col("elapsed_ns") > 0))
            .with_columns(
                (pl.col("n_calls") / (pl.col("elapsed_ns") * 1e-9)).alias("calls_per_sec"),
            )
            .with_columns(
                pl.col("n_samples").log(base=10).alias("log10_n_samples"),
                pl.col("calls_per_sec").log(base=10).alias("log10_calls_per_sec"),
                pl.col("method").replace(METHOD_LABELS).alias("method_label"),
            )
            .to_pandas()
        )
        if not n_df.empty:
            hue_order_n = (
                n_df.groupby("method_label")["calls_per_sec"]
                .max()
                .sort_values(ascending=False)
                .index.tolist()
            )
            lmplot(
                n_df,
                x_col="log10_n_samples",
                y_col="log10_calls_per_sec",
                x_label="Cohort size (N samples)",
                y_label="Throughput (alt calls / sec)",
                hue_order=hue_order_n,
                output_dir=output_dir,
                stem="n_plot",
            )

    # ---- Setup-cost plot, q-len axis only (existing behavior, full cohort) ----
    setup_df = (
        raw.filter(pl.col("n_samples") == full_n)
        .filter(
            (pl.col("n_calls") > 0)
            & pl.col("setup_ns").is_not_null()
            & (pl.col("setup_ns") > 0)
        )
        .with_columns(
            (pl.col("n_calls") / (pl.col("setup_ns") * 1e-9)).alias("setup_calls_per_sec"),
        )
        .with_columns(
            pl.col("query_length").log(base=10).alias("log10_query_length"),
            pl.col("setup_calls_per_sec").log(base=10).alias("log10_setup_calls_per_sec"),
            pl.col("method").replace(METHOD_LABELS).alias("method_label"),
        )
        .to_pandas()
    )

    if not setup_df.empty:
        setup_hue_order = (
            setup_df.groupby("method_label")["setup_calls_per_sec"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        lmplot(
            setup_df,
            x_col="log10_query_length",
            y_col="log10_setup_calls_per_sec",
            x_label="Query length (bp)",
            y_label="Setup throughput (alt calls / sec)",
            hue_order=setup_hue_order,
            output_dir=output_dir,
            stem="setup_plot",
        )
```

(Imports at the top of the file are unchanged — `_plot_common` already provides `METHOD_LABELS` and `lmplot`.)

- [ ] **Step 2: Smoke-check the script parses**

Run: `pixi r python variant_throughput/bin/plot_throughput.py --help`

Expected: cyclopts help text, exit 0.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/plot_throughput.py
git commit -m "feat(variant_throughput): plot_throughput emits N-axis figure"
```

---

## Task 7: Extend `plot_memory.py` with an N-axis figure

**Files:**
- Modify: `variant_throughput/bin/plot_memory.py`

Apply the same q-len-vs-N slicing pattern as Task 6, for the peak-RSS metric.

- [ ] **Step 1: Replace the body of `bench`**

Replace the body of `bench(...)` in `variant_throughput/bin/plot_memory.py` with:

```python
def bench(*csvs: Path, output_dir: Path = Path("results")):
    import polars as pl
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.5)

    if not csvs:
        csvs = tuple(output_dir / f"{m}_memory.csv" for m in METHOD_LABELS)

    raw = pl.concat([pl.read_csv(p) for p in csvs])

    # ---- q-len axis: full-cohort rows only ----
    full_n = raw["n_samples"].max()
    qlen_df = (
        raw.filter(pl.col("n_samples") == full_n)
        .filter((pl.col("n_calls") > 0) & (pl.col("peak_rss_bytes") > 0))
        .with_columns((pl.col("peak_rss_bytes") / 2**20).alias("peak_rss_mib"))
        .with_columns(
            pl.col("query_length").log(base=10).alias("log10_query_length"),
            pl.col("peak_rss_mib").log(base=10).alias("log10_peak_rss_mib"),
            pl.col("method").replace(METHOD_LABELS).alias("method_label"),
        )
        .to_pandas()
    )

    if not qlen_df.empty:
        hue_order = (
            qlen_df.groupby("method_label")["peak_rss_mib"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        lmplot(
            qlen_df,
            x_col="log10_query_length",
            y_col="log10_peak_rss_mib",
            x_label="Query length (bp)",
            y_label="Peak RSS (MiB)",
            hue_order=hue_order,
            output_dir=output_dir,
            stem="memory_plot",
        )

    # ---- N axis: rows whose query_length has more than one distinct n_samples ----
    n_distinct_by_q = raw.group_by("query_length").agg(
        pl.col("n_samples").n_unique().alias("n_unique_samples")
    )
    n_sweep_qlens = n_distinct_by_q.filter(pl.col("n_unique_samples") > 1)["query_length"].to_list()

    if n_sweep_qlens:
        n_df = (
            raw.filter(pl.col("query_length").is_in(n_sweep_qlens))
            .filter((pl.col("n_calls") > 0) & (pl.col("peak_rss_bytes") > 0))
            .with_columns((pl.col("peak_rss_bytes") / 2**20).alias("peak_rss_mib"))
            .with_columns(
                pl.col("n_samples").log(base=10).alias("log10_n_samples"),
                pl.col("peak_rss_mib").log(base=10).alias("log10_peak_rss_mib"),
                pl.col("method").replace(METHOD_LABELS).alias("method_label"),
            )
            .to_pandas()
        )
        if not n_df.empty:
            hue_order_n = (
                n_df.groupby("method_label")["peak_rss_mib"]
                .max()
                .sort_values(ascending=False)
                .index.tolist()
            )
            lmplot(
                n_df,
                x_col="log10_n_samples",
                y_col="log10_peak_rss_mib",
                x_label="Cohort size (N samples)",
                y_label="Peak RSS (MiB)",
                hue_order=hue_order_n,
                output_dir=output_dir,
                stem="n_memory_plot",
            )
```

- [ ] **Step 2: Smoke-check**

Run: `pixi r python variant_throughput/bin/plot_memory.py --help`

Expected: cyclopts help text, exit 0.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/bin/plot_memory.py
git commit -m "feat(variant_throughput): plot_memory emits N-axis figure"
```

---

## Task 8: Refactor `.nf` bench inputs to a `SweepInput` record

**Files:**
- Modify: `variant_throughput/variant_throughput.nf`

The 8 bench processes currently take a `Pairs` record (just `query_length` + pairs path) and read svar/bcf/pgen paths from `params`. To serve both sweeps, the same processes will receive a richer record containing the per-run file paths and the `n_samples` value.

**Reminder:** typed Nextflow 26.04 requires `record(field: Type, ...)` destructuring on inputs, not `(name): Tuple<...>`, and `output:` lines must be value-producing expressions (e.g. `record(...)` / `file(...)`), never bare identifiers.

- [ ] **Step 1: Replace the `Pairs` record and rewire bench processes**

In `variant_throughput/variant_throughput.nf`, replace the existing `record Pairs { ... }` block (currently near the bottom of the file) with:

```nextflow
record SweepInput {
    query_length: Integer
    n_samples: Integer
    pairs: Path
    svar: Path
    bcf: Path
    pgen: Path
}
```

In each of the 8 bench processes (`BENCH_SVAR_THROUGHPUT`, `BENCH_SVAR_MEMORY`, `BENCH_BCF_THROUGHPUT`, `BENCH_BCF_MEMORY`, `BENCH_PGEN_THROUGHPUT`, `BENCH_PGEN_MEMORY`, `BENCH_PRESUBSET_BCF_THROUGHPUT`, `BENCH_PRESUBSET_BCF_MEMORY`):

1. Change the input declaration from `p: Pairs` to `p: SweepInput`.
2. Replace `${params.svar}` / `${params.bcf}` / `${params.pgen}` in the `script:` block with `${p.svar}` / `${p.bcf}` / `${p.pgen}`.
3. Append `--n-samples ${p.n_samples}` to the bench command.
4. Change the output CSV filename from `..._q${p.query_length}_...csv` to `..._q${p.query_length}_n${p.n_samples}_...csv` so the q-len and N sweeps don't collide on output names.

Example (apply analogously to all 8): `BENCH_SVAR_THROUGHPUT` becomes:

```nextflow
process BENCH_SVAR_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "svar", csv: file("svar_q${p.query_length}_n${p.n_samples}_throughput.csv"))

    script:
    pack_flag = params.use_custom_pack ? "--use-custom-pack" : "--no-use-custom-pack"
    """
    bench_svar.py \\
      ${p.pairs} \\
      ${p.svar} \\
      svar_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples} \\
      ${pack_flag}
    """
}
```

Apply the same three-line edit (input type, path interpolations, CSV name, `--n-samples` flag) to:
- `BENCH_SVAR_MEMORY` — mode memory, output csv `svar_q${...}_n${...}_memory.csv`
- `BENCH_BCF_THROUGHPUT` — uses `${p.bcf}`, csv `bcf_q${...}_n${...}_throughput.csv`
- `BENCH_BCF_MEMORY` — uses `${p.bcf}`, csv `bcf_q${...}_n${...}_memory.csv`
- `BENCH_PGEN_THROUGHPUT` — uses `${p.pgen}`, csv `pgen_q${...}_n${...}_throughput.csv`
- `BENCH_PGEN_MEMORY` — uses `${p.pgen}`, csv `pgen_q${...}_n${...}_memory.csv`
- `BENCH_PRESUBSET_BCF_THROUGHPUT` — uses `${p.bcf}`, csv `presubset_bcf_q${...}_n${...}_throughput.csv`
- `BENCH_PRESUBSET_BCF_MEMORY` — uses `${p.bcf}`, csv `presubset_bcf_q${...}_n${...}_memory.csv`

Each `output:` block in these processes must use the **new csv filename in its `file(...)` expression** so it matches the script output.

Also update the existing `GENERATE_PAIRS` process **output** to emit a `SweepInput` record (so the q-len sweep wires into the new pipeline shape):

```nextflow
process GENERATE_PAIRS {
    queue 'carter-compute'
    cpus 2
    time 2.h
    memory 16.GB

    input:
    query_length: Integer
    n_replicates: Integer
    max_pairs: Integer
    seed: Integer
    max_total_length: Integer
    n_samples_full: Integer

    output:
    record(
        query_length: query_length,
        n_samples: n_samples_full,
        pairs: file("pairs_${query_length}.parquet"),
        svar: file("${params.svar}"),
        bcf: file("${params.bcf}"),
        pgen: file("${params.pgen}"),
    )

    script:
    """
    generate_pairs.py \\
      ${params.svar} \\
      ${params.fai} \\
      ${query_length} \\
      pairs_${query_length}.parquet \\
      --seed ${seed} \\
      --n-replicates ${n_replicates} \\
      --max-pairs ${max_pairs} \\
      --max-total-length ${max_total_length}
    """
}
```

Update the workflow call site to pass `n_samples_full` (a `Value<Integer>` computed in the next task). The `entry workflow`'s `main:` block currently reads:

```nextflow
lengths = channel.fromList(params.query_lengths)
pairs = GENERATE_PAIRS(lengths, params.n_replicates, params.max_pairs, params.seed, params.max_total_length)
```

Replace it with (the `n_full` value is wired in Task 9):

```nextflow
lengths = channel.fromList(params.query_lengths)
pairs = GENERATE_PAIRS(
    lengths,
    params.n_replicates,
    params.max_pairs,
    params.seed,
    params.max_total_length,
    n_full,
)
```

- [ ] **Step 2: Lint via `nextflow run -preview` smoke test (no execution)**

Run from the `variant_throughput/` directory:

```bash
cd variant_throughput && pixi r nextflow run variant_throughput.nf -preview -c configs/smoke.config 2>&1 | tail -40 && cd ..
```

Expected: the workflow parses and prints a DAG / preview without "compilation failed" or "ERROR" lines. It is OK if the preview reports that `n_full` is not yet defined — Task 9 wires it in. If the only error is about `n_full`, proceed. Any other syntax error must be fixed before committing.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/variant_throughput.nf
git commit -m "refactor(variant_throughput): bench processes take SweepInput record"
```

---

## Task 9: Add full-cohort count process and wire `n_full`

**Files:**
- Modify: `variant_throughput/variant_throughput.nf`

We need the total sample count of the full input dataset as an `Integer` value so q-len sweep rows can record their true `n_samples`. Compute it once via `make_sample_list.py --print-total`.

- [ ] **Step 1: Add `COUNT_FULL_SAMPLES` process**

Insert near the other support processes in `variant_throughput/variant_throughput.nf`:

```nextflow
process COUNT_FULL_SAMPLES {
    queue 'carter-compute'
    cpus 1
    time 30.min
    memory 8.GB

    input:
    svar: Path

    output:
    n: Integer = stdout().trim().toInteger()

    script:
    """
    make_sample_list.py ${svar} /dev/null --print-total
    """
}
```

- [ ] **Step 2: Wire `n_full` into the entry workflow**

In the entry workflow's `main:` block, add as the first statement:

```nextflow
n_full = COUNT_FULL_SAMPLES(params.svar)
```

`n_full` is now a `Value<Integer>` that flows into `GENERATE_PAIRS` (added in Task 8).

- [ ] **Step 3: Preview-lint**

```bash
cd variant_throughput && pixi r nextflow run variant_throughput.nf -preview -c configs/smoke.config 2>&1 | tail -40 && cd ..
```

Expected: no compilation errors. `n_full` is resolved.

- [ ] **Step 4: Commit**

```bash
git add variant_throughput/variant_throughput.nf
git commit -m "feat(variant_throughput): COUNT_FULL_SAMPLES wires n_full for q-len sweep"
```

---

## Task 10: Add N-sweep setup processes

**Files:**
- Modify: `variant_throughput/variant_throughput.nf`

Add four new processes that build N-specific dataset files. These are untimed and Nextflow-cached.

**Note on `bcftools`/`plink2` availability:** the existing `BENCH_*` processes assume `bcftools` and `plink2` are on `$PATH` on the carter cluster (the existing `bench_presubset_bcf.py` already shells out to `bcftools`). Reuse that assumption. Do not add a `conda` directive.

- [ ] **Step 1: Add `MAKE_SAMPLE_LIST`**

```nextflow
process MAKE_SAMPLE_LIST {
    queue 'carter-compute'
    cpus 1
    time 30.min
    memory 8.GB

    input:
    n: Integer
    seed: Integer
    svar: Path

    output:
    record(n: n, samples: file("samples_N${n}.txt"))

    script:
    """
    make_sample_list.py ${svar} samples_N${n}.txt --n ${n} --seed ${seed}
    """
}
```

- [ ] **Step 2: Add `SUBSET_BCF`**

```nextflow
process SUBSET_BCF {
    queue 'carter-compute'
    cpus 4
    time 4.h
    memory 16.GB

    input:
    n: Integer
    samples: Path
    bcf: Path

    output:
    record(n: n, bcf: file("N${n}.bcf"), csi: file("N${n}.bcf.csi"))

    script:
    """
    bcftools view -S ${samples} --force-samples --no-update --threads ${task.cpus} -Ob -o N${n}.bcf ${bcf}
    bcftools index --threads ${task.cpus} N${n}.bcf
    """
}
```

- [ ] **Step 3: Add `SUBSET_PGEN`**

```nextflow
process SUBSET_PGEN {
    queue 'carter-compute'
    cpus 4
    time 4.h
    memory 16.GB

    input:
    n: Integer
    samples: Path
    pgen: Path

    stage:
    stageAs pgen, 'in.pgen'

    output:
    record(
        n: n,
        pgen: file("N${n}.pgen"),
        pvar: file("N${n}.pvar"),
        psam: file("N${n}.psam"),
    )

    script:
    // plink2 requires its --keep file to be FID<TAB>IID; the samples.txt
    // produced by make_sample_list.py is IID-only, so prepend "0\\t" per line.
    """
    awk 'BEGIN{OFS="\\t"} {print "0", \$1}' ${samples} > keep.tsv
    pgen_stem=\$(basename in.pgen .pgen)
    # Stage sibling files (.pvar/.psam) by symlink — Nextflow stages only the
    # primary; plink2 expects all three with the same stem.
    ln -sf ${pgen.parent}/\${pgen_stem}.pvar in.pvar
    ln -sf ${pgen.parent}/\${pgen_stem}.psam in.psam
    plink2 --pfile in --keep keep.tsv --make-pgen --threads ${task.cpus} --out N${n}
    """
}
```

Note: the symlink trick is needed because Nextflow only stages the file you declare. If the upstream `params.pgen` is already accompanied by `.pvar`/`.psam` in the same directory (it is, per the 1kgp config), this works.

- [ ] **Step 4: Add `BUILD_SVAR_FROM_PGEN`**

```nextflow
process BUILD_SVAR_FROM_PGEN {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 8.h
    memory 64.GB

    input:
    n: Integer
    pgen: Path
    pvar: Path
    psam: Path

    output:
    record(n: n, svar: file("N${n}.svar"))

    script:
    """
    python - <<'PY'
from pathlib import Path
from genoray import SparseVar
SparseVar.from_pgen(Path("${pgen}"), Path("N${n}.svar"))
PY
    """
}
```

- [ ] **Step 5: Add `GENERATE_PAIRS_N` (mirrors `GENERATE_PAIRS` but for the N-sweep, with a per-N subset svar)**

```nextflow
process GENERATE_PAIRS_N {
    queue 'carter-compute'
    cpus 2
    time 2.h
    memory 16.GB

    input:
    n: Integer
    svar: Path
    bcf: Path
    pgen: Path
    query_length: Integer
    n_replicates: Integer
    max_pairs: Integer
    seed: Integer
    max_total_length: Integer

    output:
    record(
        query_length: query_length,
        n_samples: n,
        pairs: file("pairs_N${n}.parquet"),
        svar: svar,
        bcf: bcf,
        pgen: pgen,
    )

    script:
    """
    generate_pairs.py \\
      ${svar} \\
      ${params.fai} \\
      ${query_length} \\
      pairs_N${n}.parquet \\
      --seed ${seed} \\
      --n-replicates ${n_replicates} \\
      --max-pairs ${max_pairs} \\
      --max-total-length ${max_total_length}
    """
}
```

- [ ] **Step 6: Preview-lint**

```bash
cd variant_throughput && pixi r nextflow run variant_throughput.nf -preview -c configs/smoke.config 2>&1 | tail -60 && cd ..
```

Expected: no compilation errors. New processes are visible in the DAG (even though not yet called from the workflow body — Task 11 wires them in).

- [ ] **Step 7: Commit**

```bash
git add variant_throughput/variant_throughput.nf
git commit -m "feat(variant_throughput): add N-sweep setup processes"
```

---

## Task 11: Wire the N-sweep path into the entry workflow

**Files:**
- Modify: `variant_throughput/variant_throughput.nf`

- [ ] **Step 1: Add the N-sweep params**

Inside the `params { ... }` block in `variant_throughput/variant_throughput.nf`, add:

```nextflow
sample_sizes: List<Integer> = [10, 32, 100, 316, 1000, 3202]
n_sweep_query_length: Integer = 131072
sample_seed: Integer = 0
```

- [ ] **Step 2: Build the N-sweep `SweepInput` channel and mix it with the q-len channel**

In the entry workflow's `main:` block, after the existing `pairs = GENERATE_PAIRS(...)` line, insert:

```nextflow
// N-sweep path: build subset files per N (untimed), generate pairs, feed bench
n_channel = channel.fromList(params.sample_sizes)

sample_lists = MAKE_SAMPLE_LIST(n_channel, params.sample_seed, params.svar)

subset_bcf_out = SUBSET_BCF(
    sample_lists.map { r -> r.n },
    sample_lists.map { r -> r.samples },
    params.bcf,
)
subset_pgen_out = SUBSET_PGEN(
    sample_lists.map { r -> r.n },
    sample_lists.map { r -> r.samples },
    params.pgen,
)
svar_out = BUILD_SVAR_FROM_PGEN(
    subset_pgen_out.map { r -> r.n },
    subset_pgen_out.map { r -> r.pgen },
    subset_pgen_out.map { r -> r.pvar },
    subset_pgen_out.map { r -> r.psam },
)

// Join the three per-N records into one Channel<Tuple<Integer, Path, Path, Path>>
// keyed on n, then call GENERATE_PAIRS_N.
n_inputs = svar_out
    .map { r -> tuple(r.n, r.svar) }
    .join(subset_bcf_out.map { r -> tuple(r.n, r.bcf) }, by: 0)
    .join(subset_pgen_out.map { r -> tuple(r.n, r.pgen) }, by: 0)
    .map { n, svar, bcf, pgen -> tuple(n, svar, bcf, pgen) }

n_pairs = GENERATE_PAIRS_N(
    n_inputs.map { n, _s, _b, _p -> n },
    n_inputs.map { _n, s, _b, _p -> s },
    n_inputs.map { _n, _s, b, _p -> b },
    n_inputs.map { _n, _s, _b, p -> p },
    params.n_sweep_query_length,
    params.n_replicates,
    params.max_pairs,
    params.seed,
    params.max_total_length,
)

// Both sweeps feed the same bench processes.
all_inputs = pairs.mix(n_pairs)
```

**Important parallel-channel sync warning** (per `writing-typed-nextflow` callout #3): the four parallel `.map` derivations on `n_inputs` above split a single channel four ways, which is the anti-pattern the skill warns about. Fix this by passing a single record-typed channel into `GENERATE_PAIRS_N` instead. **Use this version**:

Define a small record at the bottom of the file:

```nextflow
record SubsetTriple {
    n: Integer
    svar: Path
    bcf: Path
    pgen: Path
}
```

Rewrite the wiring as:

```nextflow
n_channel = channel.fromList(params.sample_sizes)

sample_lists = MAKE_SAMPLE_LIST(n_channel, params.sample_seed, params.svar)

subset_bcf_out  = SUBSET_BCF (sample_lists.map { r -> r.n }, sample_lists.map { r -> r.samples }, params.bcf)
subset_pgen_out = SUBSET_PGEN(sample_lists.map { r -> r.n }, sample_lists.map { r -> r.samples }, params.pgen)
svar_out        = BUILD_SVAR_FROM_PGEN(
    subset_pgen_out.map { r -> r.n },
    subset_pgen_out.map { r -> r.pgen },
    subset_pgen_out.map { r -> r.pvar },
    subset_pgen_out.map { r -> r.psam },
)

triples = svar_out
    .map { r -> tuple(r.n, r.svar) }
    .join(subset_bcf_out.map { r -> tuple(r.n, r.bcf) },  by: 0)
    .join(subset_pgen_out.map { r -> tuple(r.n, r.pgen) }, by: 0)
    .map { n, svar, bcf, pgen ->
        record(n: n, svar: svar, bcf: bcf, pgen: pgen) as SubsetTriple
    }
```

And change `GENERATE_PAIRS_N`'s input section (in Task 10) so it accepts the record. **Apply this edit now** to the `GENERATE_PAIRS_N` process you added in Task 10:

```nextflow
input:
t: SubsetTriple
query_length: Integer
n_replicates: Integer
max_pairs: Integer
seed: Integer
max_total_length: Integer

output:
record(
    query_length: query_length,
    n_samples: t.n,
    pairs: file("pairs_N${t.n}.parquet"),
    svar: t.svar,
    bcf: t.bcf,
    pgen: t.pgen,
)

script:
"""
generate_pairs.py \\
  ${t.svar} \\
  ${params.fai} \\
  ${query_length} \\
  pairs_N${t.n}.parquet \\
  --seed ${seed} \\
  --n-replicates ${n_replicates} \\
  --max-pairs ${max_pairs} \\
  --max-total-length ${max_total_length}
"""
```

Then call it as:

```nextflow
n_pairs = GENERATE_PAIRS_N(
    triples,
    params.n_sweep_query_length,
    params.n_replicates,
    params.max_pairs,
    params.seed,
    params.max_total_length,
)

all_inputs = pairs.mix(n_pairs)
```

- [ ] **Step 3: Point the 8 existing bench process calls at `all_inputs`**

The current workflow has, after `pairs = GENERATE_PAIRS(...)`:

```nextflow
svar_t = BENCH_SVAR_THROUGHPUT(pairs)
bcf_t  = BENCH_BCF_THROUGHPUT(pairs)
...
```

Replace every `pairs` argument in those eight bench calls with `all_inputs`.

- [ ] **Step 4: Preview-lint**

```bash
cd variant_throughput && pixi r nextflow run variant_throughput.nf -preview -c configs/smoke.config 2>&1 | tail -60 && cd ..
```

Expected: no compilation errors. DAG shows the N-sweep setup chain feeding into the bench processes alongside the q-len chain.

- [ ] **Step 5: Commit**

```bash
git add variant_throughput/variant_throughput.nf
git commit -m "feat(variant_throughput): wire N-sweep into entry workflow"
```

---

## Task 12: Smoke-test with `configs/smoke.config`

The repo already contains `variant_throughput/configs/smoke.config`. We will run the pipeline end-to-end against it to verify both sweep paths produce CSVs and plots without execution-time errors. Smoke config is expected to be small enough to finish in minutes.

- [ ] **Step 1: Inspect smoke config to confirm it sets small sample_sizes and a single query length**

Run: `pixi r cat variant_throughput/configs/smoke.config`

If it does not override `sample_sizes` to something small (e.g. `[2, 5]`), `n_sweep_query_length` to something cheap (e.g. `8192`), and `query_lengths` to a short list, edit it to add those overrides. Append (or merge into the existing `params { ... }`):

```groovy
params {
    sample_sizes         = [2, 5]
    n_sweep_query_length = 8192
    query_lengths        = [4096, 8192]
    n_replicates         = 2
    max_pairs            = 4
    max_total_length     = 65536
}
```

(Only add keys that are not already set; do not duplicate.)

- [ ] **Step 2: Run the pipeline**

```bash
cd variant_throughput && pixi r nextflow run variant_throughput.nf -c configs/smoke.config -resume 2>&1 | tee smoke.log | tail -80 && cd ..
```

Expected: pipeline completes with `Succeeded` non-zero count and no `FAILED` processes. Output directory should contain `svar_throughput.csv`, `bcf_throughput.csv`, `pgen_throughput.csv`, `presubset_bcf_throughput.csv` plus the four `_memory.csv` analogues, and both `plot.{png,svg,pdf}` and `n_plot.{png,svg,pdf}` (and same for memory).

- [ ] **Step 3: Verify each combined CSV has the new `n_samples` column with both q-len and N values**

```bash
pixi r python -c "
import polars as pl, pathlib
for f in pathlib.Path('variant_throughput/results').glob('*_throughput.csv'):
    df = pl.read_csv(f)
    print(f.name, 'columns:', df.columns)
    print('  n_samples values:', sorted(df['n_samples'].unique().to_list()))
    print('  query_length values:', sorted(df['query_length'].unique().to_list()))
"
```

Expected: every CSV lists `n_samples` in `columns`, and `n_samples` has at least two distinct values (the full cohort count from the q-len sweep plus the N-sweep values). `query_length` has the smoke `query_lengths` values plus `n_sweep_query_length`.

- [ ] **Step 4: Commit any smoke-config edits**

```bash
git add variant_throughput/configs/smoke.config
git commit -m "test(variant_throughput): smoke config exercises N sweep"
```

---

## Task 13: Wire N-sweep plots into the workflow `output` block

**Files:**
- Modify: `variant_throughput/variant_throughput.nf`

The plot scripts now emit `n_plot.{png,svg,pdf}` and `n_memory_plot.{png,svg,pdf}` alongside the existing artifacts. Publish them.

- [ ] **Step 1: Extend `PLOT_THROUGHPUT` and `PLOT_MEMORY` outputs**

In `variant_throughput/variant_throughput.nf`, update the `PLOT_THROUGHPUT` process output to include the N-axis files:

```nextflow
output:
record(
    plot_png: file("plot.png"),
    plot_svg: file("plot.svg"),
    plot_pdf: file("plot.pdf"),
    setup_png: file("setup_plot.png"),
    setup_svg: file("setup_plot.svg"),
    setup_pdf: file("setup_plot.pdf"),
    n_plot_png: file("n_plot.png", optional: true),
    n_plot_svg: file("n_plot.svg", optional: true),
    n_plot_pdf: file("n_plot.pdf", optional: true),
)
```

(`optional: true` so smoke runs without an N-sweep variant don't fail.)

Update the `ThroughputPlots` record at the bottom of the file:

```nextflow
record ThroughputPlots {
    plot_png: Path
    plot_svg: Path
    plot_pdf: Path
    setup_png: Path
    setup_svg: Path
    setup_pdf: Path
    n_plot_png: Path?
    n_plot_svg: Path?
    n_plot_pdf: Path?
}
```

Update the workflow's `output { throughput_plots: ... }` block:

```nextflow
throughput_plots: Value<ThroughputPlots> {
    path { r ->
        r.plot_png >> "plot.png"
        r.plot_svg >> "plot.svg"
        r.plot_pdf >> "plot.pdf"
        r.setup_png >> "setup_plot.png"
        r.setup_svg >> "setup_plot.svg"
        r.setup_pdf >> "setup_plot.pdf"
        if (r.n_plot_png) r.n_plot_png >> "n_plot.png"
        if (r.n_plot_svg) r.n_plot_svg >> "n_plot.svg"
        if (r.n_plot_pdf) r.n_plot_pdf >> "n_plot.pdf"
    }
}
```

Same shape for `PLOT_MEMORY` — add `n_memory_plot.{png,svg,pdf}` to the process output (with `optional: true`), to the `MemoryPlots` record (as `Path?` fields), and to the workflow output block.

`PLOT_MEMORY` output:

```nextflow
output:
record(
    png: file("memory_plot.png"),
    svg: file("memory_plot.svg"),
    pdf: file("memory_plot.pdf"),
    n_png: file("n_memory_plot.png", optional: true),
    n_svg: file("n_memory_plot.svg", optional: true),
    n_pdf: file("n_memory_plot.pdf", optional: true),
)
```

`MemoryPlots` record:

```nextflow
record MemoryPlots {
    png: Path
    svg: Path
    pdf: Path
    n_png: Path?
    n_svg: Path?
    n_pdf: Path?
}
```

Memory output block:

```nextflow
memory_plots: Value<MemoryPlots> {
    path { r ->
        r.png >> "memory_plot.png"
        r.svg >> "memory_plot.svg"
        r.pdf >> "memory_plot.pdf"
        if (r.n_png) r.n_png >> "n_memory_plot.png"
        if (r.n_svg) r.n_svg >> "n_memory_plot.svg"
        if (r.n_pdf) r.n_pdf >> "n_memory_plot.pdf"
    }
}
```

- [ ] **Step 2: Re-run the smoke pipeline and confirm N-axis plot files appear in the published `results/` directory**

```bash
cd variant_throughput && pixi r nextflow run variant_throughput.nf -c configs/smoke.config -resume 2>&1 | tail -40 && cd ..
ls variant_throughput/results/ | grep -E '^(n_plot|n_memory_plot)\.'
```

Expected: `n_plot.png`, `n_plot.svg`, `n_plot.pdf`, `n_memory_plot.png`, `n_memory_plot.svg`, `n_memory_plot.pdf` are all listed.

- [ ] **Step 3: Commit**

```bash
git add variant_throughput/variant_throughput.nf
git commit -m "feat(variant_throughput): publish N-axis plots"
```

---

## Task 14: Final preview against the 1kGP config

- [ ] **Step 1: Preview against the real 1kgp config**

```bash
cd variant_throughput && pixi r nextflow run variant_throughput.nf -preview -c configs/1kgp.config 2>&1 | tail -60 && cd ..
```

Expected: the preview lists `MAKE_SAMPLE_LIST`, `SUBSET_BCF`, `SUBSET_PGEN`, `BUILD_SVAR_FROM_PGEN`, `GENERATE_PAIRS_N` once per element of `sample_sizes` (6 elements at default), `COUNT_FULL_SAMPLES` once, and each of the 8 bench processes scheduled `len(sample_sizes) + len(query_lengths)` times. No compilation errors.

If counts look wrong, return to Task 11 and re-verify the channel wiring.

- [ ] **Step 2: Stop here**

Do not launch the full 1kGP run from this plan — that's a multi-day cluster job and should be initiated by the user. The plan is complete once the preview succeeds.

---

## Self-Review Notes

- **Spec coverage:**
  - New params (`sample_sizes`, `n_sweep_query_length`, `sample_seed`): Task 11.
  - Per-N untimed setup chain (`MAKE_SAMPLE_LIST`, `SUBSET_BCF`, `SUBSET_PGEN`, `BUILD_SVAR_FROM_PGEN`): Task 10.
  - Pairs regenerated per N against the subset svar: `GENERATE_PAIRS_N` (Tasks 10–11).
  - `n_samples` column emitted by every bench script: Tasks 1–4.
  - Full-cohort passthrough when `n` equals the full count: Task 5 (`make_sample_list.py` handles it via the `n >= len(available)` branch); the bench files still get rebuilt per N — that's a conscious deviation from the spec's "symlink passthrough" optimization, in favor of simplicity. If the smoke test in Task 12 shows this is a problem, optimize then.
  - Combined CSVs across both sweeps: existing `COMBINE_*` processes work unchanged because `BENCH_*` output records already carry per-method CSVs into them.
  - Per-axis plots: Tasks 6, 7, 13.
- **Placeholder scan:** No `TBD`/`TODO`/"appropriate" placeholders. All code blocks are concrete.
- **Type consistency:** `SweepInput` fields are used identically across `GENERATE_PAIRS`, `GENERATE_PAIRS_N`, and all eight `BENCH_*` processes. `SubsetTriple` is the only intermediate record and is used in exactly one place (`GENERATE_PAIRS_N`'s input).
- **Deviation from spec, called out:** spec specifies "symlink/passthrough" when `n` is the full cohort; the plan instead rebuilds the subset files even at full N. Rationale: it's one fewer special-case branch in the typed Nextflow wiring, and the rebuild is one-time and cached. Full-N is implicitly already covered by the q-len sweep, so an N-sweep point at full N is somewhat redundant; the user can leave the trailing `3202` in `sample_sizes` or drop it.
