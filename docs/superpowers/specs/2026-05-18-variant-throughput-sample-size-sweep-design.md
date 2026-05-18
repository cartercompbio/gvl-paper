# Variant throughput / memory: sample-size sweep over 1kGP

## Goal

Add a sample-size (N) sweep to `variant_throughput/variant_throughput.nf` so the
existing throughput and peak-RSS benchmarks can be evaluated as a function of
cohort size, in addition to the current query-length sweep. Target dataset is
1kGP (~3202 samples).

## Scope

- Add an **orthogonal** N sweep at a single fixed query length. Existing
  query-length sweep (full N) is untouched.
- Apply to all four methods already in the pipeline: `svar`, `bcf`, `pgen`,
  `presubset_bcf`.
- Both `throughput` and `memory` modes for each method.
- Subsetting cost is **not** measured — only the existing timed/sampled regions
  in `bench_*.py` are reported.

Out of scope: changing the existing bench scripts' timed regions, adding new
formats, full N × query-length cross-product.

## Sweep configuration

New `params` in `variant_throughput.nf`:

| Name | Type | Default | Meaning |
|---|---|---|---|
| `sample_sizes` | `List<Integer>` | `[10, 32, 100, 316, 1000, 3202]` | N values for the sweep (log-spaced). |
| `n_sweep_query_length` | `Integer` | `131072` | Fixed query length used for the N sweep. |
| `sample_seed` | `Integer` | `0` | RNG seed for selecting which samples land in each N subset. Distinct from `seed` (which seeds pair sampling) so changing one doesn't shift the other. |

`sample_sizes` may include the full cohort count (3202 for 1kGP); the workflow
must short-circuit subsetting in that case and reuse the input files directly.

## Architecture

Two parallel sub-workflows feed a shared combine/plot stage:

```
                  q_len_sweep  ──┐
                                 ├── COMBINE_* per method ── PLOT_*
                  n_sweep      ──┘
```

The q-len sweep is the current workflow, unmodified.

The N sweep adds, per `n ∈ sample_sizes`:

1. `MAKE_SAMPLE_LIST(n)` → `samples_N{n}.txt` (one sample ID per line).
2. `SUBSET_BCF(n, samples_N{n}.txt)` → `N{n}.bcf` + `.csi`.
3. `SUBSET_PGEN(n, samples_N{n}.txt)` → `N{n}.{pgen,pvar,psam}`.
4. `BUILD_SVAR_FROM_PGEN(N{n}.pgen)` → `N{n}.svar`.
5. `GENERATE_PAIRS_N(n, N{n}.svar, q_len = n_sweep_query_length)` → pairs parquet.
6. The eight existing bench processes run against `(N{n}.svar, N{n}.bcf, N{n}.pgen, pairs_N{n})`.

If `n` equals the full sample count, steps 2–4 become symlinks/passthroughs to
the input `params.{bcf,pgen,svar}` to avoid wasted work.

`SUBSET_BCF` uses `bcftools view -S samples.txt -Ob` followed by `bcftools index`.
`SUBSET_PGEN` uses `plink2 --pfile ... --keep samples.txt --make-pgen`.
`BUILD_SVAR_FROM_PGEN` uses `genoray.SparseVar.from_pgen` (confirmed available
in the `bench` pixi env).

### Sample selection

`MAKE_SAMPLE_LIST` reads `_svar.available_samples`, shuffles deterministically
with `sample_seed`, and takes the first `n`. Subsets are **nested by
construction** (sorting after shuffle would break this; we don't sort).
Nesting isn't required for correctness, but it makes the N curves slightly
more interpretable.

### Pairs for the N sweep

Pairs are regenerated per N against the subset svar, at fixed
`query_length = n_sweep_query_length`. This means each (N) replicate draws its
samples from that subset's available pool — consistent with the rest of the
sweep's "this format has N samples" framing. Query regions will differ across N
(since RNG state diverges), which is acceptable: each N has `n_replicates`
replicates, so variance across query regions is averaged within each N point.

## Output schema

Each `bench_*.py` invocation already writes a CSV with columns:

`dataset, method, query_length, replicate, n_pairs, n_calls, elapsed_ns,
setup_ns` (throughput mode) or `peak_rss_bytes` (memory mode).

Add one column: `n_samples: Integer`.

- For q-len-sweep rows: `n_samples` = the full cohort count of the input file.
- For N-sweep rows: `n_samples` = the N value used to build the subset.

Wiring: pass `--n-samples ${n}` (and the same flag with the full count for the
q-sweep path) through the nf script into each `bench_*.py`. Each script writes
the value into every row. No other timed-region changes.

`COMBINE_THROUGHPUT` / `COMBINE_MEMORY` continue to concatenate per-method CSVs
across both sweeps. Downstream consumers can filter by
`query_length == n_sweep_query_length` vs `n_samples == full_count` to recover
each 1-D sweep.

## Plotting

Extend the existing plot scripts so they emit, in addition to the current
"throughput vs query_length" / "memory vs query_length" figures, one figure per
mode against `n_samples`. Implementation: each script detects two slices in the
combined CSV (q-sweep rows vs N-sweep rows) and writes two sets of artifacts.

New output records in `variant_throughput.nf`:

- `n_throughput_plots` — `plot.{png,svg,pdf}`, `setup_plot.{png,svg,pdf}`
- `n_memory_plots` — `memory_plot.{png,svg,pdf}`

Existing `throughput_plots` / `memory_plots` outputs remain.

## Resource notes

- N-sweep total work ≈ 6 N × 4 methods × 2 modes × 5 replicates at q_len=131072.
  Each replicate is small at this q_len, so this is a fraction of the existing
  q-sweep cost.
- Subset materialization is one-time per (N, format) and Nextflow-cached. The
  full N=3202 passthrough costs nothing.
- All subset processes run on `carter-compute` matching the surrounding
  benchmark processes; `SUBSET_PGEN` and `SUBSET_BCF` are CPU-light; the
  `BUILD_SVAR_FROM_PGEN` step is the heaviest of the three and should get a
  similar `cpus`/`memory` budget to `BENCH_SVAR_*`.

## Files touched

- `variant_throughput/variant_throughput.nf` — new params, new processes, new
  publish outputs.
- `variant_throughput/bin/generate_pairs.py` — no change (already takes svar
  path and query length).
- `variant_throughput/bin/bench_{svar,bcf,pgen,presubset_bcf}.py` — accept and
  emit `--n-samples`.
- `variant_throughput/bin/plot_throughput.py`,
  `variant_throughput/bin/plot_memory.py` — add N-axis plots.
- `variant_throughput/bin/` — new helper script for `MAKE_SAMPLE_LIST` if a
  small Python script is cleaner than an inline `nf` block (likely yes).
- `variant_throughput/configs/1kgp.config` — no change required; defaults are
  set in the workflow.

## Open / deferred

- The 1kGP `available_samples` count is assumed to be 3202; the workflow
  should assert that the largest entry in `sample_sizes` does not exceed the
  cohort, but won't otherwise validate.
- No GDC / TCGA / UKBB sample-size sweep in this change.
