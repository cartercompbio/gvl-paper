# GVL 0.27.0 full throughput + memory bench (SVAR-backed) — design

**Date:** 2026-06-05
**Branch:** `feat/gvl-026-parity-probe` (continues the gvl027 line)
**Status:** approved design, pending spec review

## Goal

Confirm GenVarLoader **0.27.0** matches or beats **0.6.1** on the full
manuscript throughput grid, and characterize its memory use, across **three
datasets** (tcga-atac, 1kgp, ukbb) for **haplotype and track** dataloading.
All GVL datasets are **SVAR-backed only**. genoray is bumped to **2.9.0** and
the new `--no-symbolic` / `--no-breakend` filters (and their Python
equivalents) are applied wherever a VCF/PGEN is read into an SVAR.

Unlike the earlier reduced *probe*, this is the full `(threads x batch_size)`
grid run through the production Nextflow harness (`benchmark.nf`), measuring
**throughput** (default + single-`buffered` dataloading) and **peak/avg RSS
memory**.

## Scope

In scope:
- Full grid (`make_launch_grid.py`: threads 1->64, batch-size sweep) at seqlens
  `[2048, 16384, 131072, 1048576]`, run via `benchmark.nf` under genvarloader
  0.27.0 (pixi `bench027` env).
- Datasets: **1kgp** (PGEN->SVAR), **tcga-atac** (BCF->SVAR, + track bench),
  **ukbb** (chr22 imputed PGEN->SVAR).
- Two dataloading modes per cell: **`none`** (default `to_dataloader`) and
  **`buffered`** (single buffered loader, `buffer_bytes = 2 GiB`).
- Throughput **and** memory passes (memory via `--measure_memory`).
- genoray 2.9.0 symbolic/breakend filtering at the SVAR-conversion step, plus a
  code-only fix to `bench_svar_vcf_plink.py`'s `VCF()`/`PGEN()` construction.

Out of scope (non-goals):
- No VCF/PGEN-backed GVL datasets (SVAR only). `native` write path stays in the
  code but is gated off by default.
- `bench_svar_vcf_plink.py` is **not re-run** (correctness fix only).
- `ref` / `bigwig` benchmarks excluded; strictly haps + tracks.
- GDC dataset excluded.
- No 0.6.1 **memory** baseline (none exists); memory reported as 0.27.0
  absolute. The throughput-vs-0.6.1 comparison is an **internal** sanity check,
  **not** a paper deliverable.

## Orchestration: `benchmark.nf` (Nextflow, typed DSL2)

The existing pipeline already provides: `BENCH_SVAR_CONVERT -> BENCH_WRITE_DATASET
-> BENCH_HAPS / BENCH_TRACKS`; per-length grid generation; a top-level
`--measure_memory` switch that flips every process into RSS mode and redirects
outputs to `*_memory/` dirs; `--region` for ukbb chr22; OOM-retry with growing
memory; and `-resume`-based caching/reuse.

A throughput run and a memory run are two invocations of the pipeline (toggling
`--measure_memory`), as today.

### Changes to `benchmark.nf`

1. **SVAR-only.** Add `bench_native: Boolean = false`. Build `all_write_inputs`
   from `svar_write_inputs` only (gate the `native_write_inputs` branch behind
   `bench_native`). Every GVL dataset is therefore SVAR-backed. Because the
   pipeline always (re)converts the SVAR itself with the correct filters,
   SVAR provenance is moot and `-resume` handles reuse — no pre-existing-SVAR
   validation needed.

2. **`dl_mode` dimension.** Introduce `dl_modes = channel.fromList(["none",
   "buffered"])`. Cross it into the haps/track inputs so each
   `(length, backend, dl_mode)` is a separate process invocation and CSV.
   Thread `--dl-mode` through `BENCH_HAPS` / `BENCH_TRACKS`. Output filenames
   and the `output {}` block gain a `_${dl_mode}` segment.

3. **genoray filter params.** Add `no_symbolic: Boolean = true`,
   `no_breakend: Boolean = true`. Pass as flags to `benchmark_svar_convert.py`
   in `BENCH_SVAR_CONVERT`.

4. **`results_dir`** points at the repo-level `results_gvl027/`.

### Env wiring (fresh `gvl027` profile)

Nextflow processes run as fresh SLURM jobs, so each must activate the
`bench027` pixi env for the Python tools. Add a `gvl027` profile (in a
`nextflow.config` under `hap_track_throughput/`) that sets
`process.beforeScript` to activate the `bench027` env (e.g. via
`pixi shell-hook -e bench027` / running tools through `pixi run -e bench027`).

**Nextflow itself is NOT added to pixi/conda** (conda Nextflow has Java-discovery
issues). The pipeline is launched with the user's local Nextflow install,
selecting `-profile gvl027`.

## genoray 2.9.0 filtering

2.9.0 adds, for `genoray write`, the CLI flags `--no-symbolic` (drop ALT with a
symbolic allele, e.g. `<DEL>`) and `--no-breakend` (drop BND records), plus the
filter expressions `genoray.exprs.is_symbolic` and `genoray.exprs.is_breakend`.

Python equivalents (verified against installed 2.9.0 source during impl):
- `PGEN(filter = ~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend)`
- `VCF(pl_filter = ~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend)`

### `benchmark_svar_convert.py`

Add `no_symbolic: bool = True`, `no_breakend: bool = True`. Build the combined
filter expression from the requested flags and pass it to the `genoray.PGEN` /
`genoray.VCF` constructor before `SparseVar.from_{pgen,vcf}`. This guarantees a
GVL-compatible SVAR (no symbolic/breakend ALTs reaching haplotype buffers).

### `bench_svar_vcf_plink.py` (correctness only, not re-run)

Apply the same `filter=` / `pl_filter=` to its `VCF(...)` and `PGEN(...)`
construction so the file remains correct under genoray 2.9.0. No execution.

## Throughput schema reconciliation

The 0.6.1 baseline `results/{hap,track}_results.csv` has columns
`dataset,threads,seqlen,batch_size,throughput (MiB/s)`. The current
`bin/benchmark_haps.py` / `benchmark_tracks.py` emit
`...,n_batches_measured,duration` (no bytes, no MiB/s) — not directly
comparable.

**Fix:** make both scripts (throughput mode) accumulate real batch bytes and
emit a row with **all of**:

```
dataset,backend,dl_mode,threads,seqlen,batch_size,n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)
```

- `total_bytes` (int) and `duration_ns` (int) are kept for precision;
  `throughput (MiB/s)` is derived (`total_bytes / 2**20 / (duration_ns/1e9)`).
- Reuse the probe's `n_bytes` / `mib_per_s` helpers (lift `_probe_common.py`
  into `bin/` or import shared). Buffered + default share the same accounting.

Memory mode schema gains `dl_mode`:

```
dataset,backend,dl_mode,threads,seqlen,batch_size,avg_rss_bytes,peak_rss_bytes
```

### Dataloading-mode handling in the scripts

- **buffered haps** requires `with_settings(deterministic=True)` (probe finding).
- buffered construction can raise `ValueError` when a single minibatch exceeds
  `buffer_bytes`; record that cell as NaN and continue (probe behavior).
- **Empty-epoch guard:** if an epoch yields 0 batches (e.g. `batch_size >
  n_instances` interacting with buffered `drop_last`), record NaN and break —
  never spin the `while not done` loop forever (root cause of the 0.26.0 hang;
  add the guard defensively even though 0.27.0 fixed the underlying behavior).

## Outputs and comparison

Nextflow `output {}` layout under `results_gvl027/`:
- Throughput: `haps/{dataset}_{length}_{backend}_{dl_mode}.csv`,
  `tracks/{dataset}_{length}_{backend}_{dl_mode}.csv`
- Memory: `haps_memory/...`, `tracks_memory/...`
- SVAR convert + dataset write benches: `svar_convert/`, `write/` (+ `_memory`).

`compare_to_baseline.py` (adapted from the probe):
- Read the new nf layout (mode from directory, `dl_mode` from filename/column).
- **Throughput, internal sanity:** join `dl_mode == "none"` rows to
  `results/{hap,track}_results.csv` on `dataset,threads,seqlen,batch_size`;
  report median 0.27.0/0.6.1 ratio + parity scatter.
- **Buffered:** reported standalone (no 0.6.1 equivalent).
- **Memory:** reported absolute — per-`(dataset,mode,dl_mode,seqlen,batch_size)`
  peak/avg RSS table + peak-RSS-vs-batch plot. No baseline join.

## Environment / deps

- Pin `genoray ==2.9.0` explicitly in `feature.bench027` (currently transitive)
  and in `feature.bench` / default (so `bench_svar_vcf_plink.py` imports a
  consistent API). Verify the pixi lock resolves.

## Verification

- **Smoke test** before the full launch: `-profile gvl027 --test_grid` (tiny
  grid) for one dataset, both `dl_mode`s, confirming the env activates, SVAR
  conversion applies the filters, `gvl.write(variants=<svar>)` succeeds, and
  haps/tracks scripts emit the new schema with no NaN on the small cells.
- Full run: every populated cell has finite throughput (NaN only where buffered
  legitimately skips); memory tables populated; report median throughput ratios
  (internal) per `(dataset, mode)` and absolute peak RSS.
- `bench_svar_vcf_plink.py` imports and a tiny smoke read confirm the updated
  genoray construction works (without running the full bench).

## Risks / open items

- **genoray filter wiring**: exact kwarg (`filter` vs `pl_filter`) per reader
  and whether `~is_symbolic` composes without an explicit `import polars` — to
  confirm against installed 2.9.0 source at impl time.
- **Env activation inside SLURM**: the `beforeScript` activation mechanism must
  work on the compute nodes; validate in the smoke test.
- **buffered on tcga-atac** large batch sizes (6200 instances): the empty-epoch
  guard must prevent any hang; explicitly exercised by the smoke/full run.
- Run cost: 3 datasets x {haps[,tracks]} x 2 dl_modes x 4 seqlens x full grid,
  twice (throughput + memory). Nextflow fan-out + `maxForks`/retry handle it;
  carter-cn-04 pin + cpus=64 per the many-core constraint.
