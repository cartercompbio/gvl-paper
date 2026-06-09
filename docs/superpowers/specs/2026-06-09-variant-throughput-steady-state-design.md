# Variant-throughput benchmark: steady-state streaming redesign

**Date:** 2026-06-09
**Scope:** `variant_throughput/` benchmark harness (SVAR vs BCF vs PGEN vs PRESUB-BCF)
**Status:** design approved, pending implementation plan

## Problem

The current benchmark times a single shot of each replicate's genotype gather.
For SVAR the gather is sub-millisecond, so the measured time is dominated by a
near-constant **~10 ms fixed cost that lands on roughly half the runs** — the
numba `parallel=True` thread-team wakeup (and, secondarily, mmap first-touch),
incurred because the harness invokes the gather once per replicate with
substantial Python work in between, letting the thread pool park.

Evidence (from `results/svar_throughput.csv`): `elapsed_ns` is bimodal — a fast
cluster ~0.15–0.28 ms and a slow cluster ~10–16 ms — and the slow value is flat
regardless of `n_calls` (50 calls and 26,581 calls both ~10.6 ms). Identical work
(same pairs, same `n_calls`) lands in either cluster, ~58× apart. This produces:

- `plot.png`: SVAR scatter spans 2–3 orders of magnitude and looks bimodal.
- `n_plot.png`: SVAR appears non-monotonic in cohort size — but per-`(region,
  sample)` call count is ~N-independent, so the lowess "hump" is purely an
  artifact of how many of 5 single-shot replicates per N caught the ~10 ms penalty.

The other three formats read dense representations whose runtime is tens of ms to
seconds, so the ~10 ms cost is negligible and they scatter tightly.

## Use-case framing

The benchmark exists to compare file formats for the **GenVarLoader training
dataloader** use case: pulling genotypes for random `(region, sample)` pairs,
batch after batch, for an entire epoch.

- The numba thread pool stays **hot** under continuous demand; the ~10 ms wakeup
  is **not** paid per batch in real training. Charging it to every query
  overstates SVAR's cost and is unrepresentative.
- GenVarLoader knows all queries ahead of time and **caches the variant-index
  search**. The only per-batch work during training is the genotype **gather**.
  So the headline throughput must be **gather only**; the index search is a
  one-time, ahead-of-time (AOT) cost.
- Stores fit in RAM on the benchmark nodes, so steady state is **compute-bound**;
  the cold mmap-fault / out-of-core regime is a separate dataset-size question
  (affecting every mmap-backed format equally) and is explicitly out of scope here.

Therefore the correct measurement is **steady-state, hot-pool gather throughput**,
realized by streaming fresh random pairs in a sustained loop — the dataloader
keeps the pool hot, so the benchmark does too.

## Design

### 1. Pair stream and batching

`generate_pairs.py` continues to sample reproducible, seeded random
`(region, sample)` pairs with the existing gap-masked, length-weighted region
sampling. Changes:

- Emit a **fixed-length stream** instead of a random `1..max_pairs` count.
- Batch the stream at `batch_size = max(1, bp_budget // query_length)` where
  `bp_budget = 2**24` (16 MiB) — the existing `max_total_length`, i.e. the
  "hold base-pairs-per-batch approximately constant" convention already used by
  the memory-growth figure. Add a `batch_id` column.
- Stream length = `stream_batches` distinct batches per replicate (default 64).
  The same stream (same parquet) is handed to **all four formats** so the
  comparison is on identical pairs.
- Keep ~5 replicates; each replicate is an **independent stream** (distinct seed)
  → one steady-state throughput point each (preserves scatter / error bars).

Parquet schema: `replicate, batch_id, contig, start, end, sample`.

### 2. Per-format AOT cache + timed replay

| Format | AOT step (once, → `setup_ns`) | Timed sustained loop (→ `elapsed_ns`) |
|---|---|---|
| SVAR | `_svar_search` per batch → cache `flat_starts/flat_ends` | `_svar_pack` over cached indices |
| BCF | — (no separable cache) | `genoray` `read` (locate+read) |
| PGEN | — | `genoray` `read` |
| PRESUB-BCF | `bcftools` subset → temp BCFs (cache paths) | `cyvcf2` read of temp BCFs |

The asymmetry — SVAR replays cached gathers while the baselines re-locate every
read — is the GenVarLoader advantage the benchmark is designed to demonstrate.

### 3. Measurement loop (shared helper `_streaming.py`)

For each cell `(query_length, n_samples, method)`:

1. Read the stream, group into batches.
2. **Outside timing:** run each format's AOT step and pre-convert every batch to
   that format's ready-to-gather inputs (cached in memory).
3. **Warmup:** execute the gather over a few batches untimed (primes thread pool,
   JIT, page cache).
4. **Sustained timed loop:** cycle the cached batches, executing only the gather,
   accumulating `n_calls`:
   ```
   t0 = perf_counter_ns()
   iters = 0
   while (perf_counter_ns() - t0) < min_ns or iters < min_batches:
       gather(batches[iters % len(batches)])
       iters += 1
   total_ns = perf_counter_ns() - t0
   ```
   `min_seconds = 5`, `min_batches = 10`. Inter-batch work is list indexing only,
   so the pool never parks. The loop yields a steady-state rate
   `rate = total_timed_calls / total_ns`.
5. **Normalize to one pass over the distinct stream** so a single `n_calls`
   column stays consistent across both the throughput and setup panels (the loop
   cycles batches, so `total_timed_calls` includes repeats and must not be
   reported directly). Report:
   - `n_calls` = calls over the **distinct** stream (no repeats).
   - `elapsed_ns` = `round(n_calls / rate)` — the single-pass-equivalent gather
     time at the measured steady-state rate, so `n_calls / elapsed_ns == rate`.
   - `setup_ns` = the AOT step's time over the **distinct** stream (single
     measurement), pairing with the same distinct `n_calls`; `None` for BCF/PGEN.
   - `n_pairs` = distinct stream pairs (`stream_batches * batch_size`).
   One row per replicate. Both `calls_per_sec` (throughput plot) and
   `setup_calls_per_sec` (setup plot) are then `n_calls / {elapsed,setup}_ns`
   against a single consistent `n_calls`.

### 4. Memory mode

Same streaming structure: AOT + warmup outside the sampler, then peak RSS
(`PeakRssSampler`) over the sustained gather loop — more representative than the
current single-batch peak.

### 5. CSV schema (unchanged, for plot compatibility)

Throughput: `dataset, method, query_length, n_samples, replicate, n_pairs,
n_calls, elapsed_ns, setup_ns`. (`n_pairs` = distinct stream pairs; `n_calls`,
`elapsed_ns`, `setup_ns` are normalized to one pass over the distinct stream — see
Measurement step 5.)
Memory: `..., n_calls, peak_rss_bytes`.
`plot_throughput.py` / `plot_memory.py` and `_plot_common.py` are unchanged.

## Files

- **`bin/generate_pairs.py`** — fixed-length stream, `batch_size` from bp budget,
  `batch_id` column; drop random `max_pairs` count.
- **`bin/_streaming.py`** (new) — shared `run_stream(gather, batches, min_seconds,
  min_batches) -> (total_calls, total_ns)` and a peak-RSS variant.
- **`bin/bench_svar.py`** — AOT `_svar_search` per batch (timed once → `setup_ns`),
  cache indices, sustained `_svar_pack` replay via `_streaming`.
- **`bin/bench_bcf.py`, `bin/bench_pgen.py`** — sustained `read` replay; no AOT.
- **`bin/bench_presubset_bcf.py`** — AOT `bcftools` subset (→ `setup_ns`),
  sustained `cyvcf2` read replay of cached temp BCFs.
- **`variant_throughput.nf`** — new params `min_seconds`, `min_batches`,
  `stream_batches`; drop `max_pairs`; thread params into `GENERATE_PAIRS[_N]` and
  the eight `BENCH_*` processes.
- **Plotting / `configs/`** — unchanged.

## Out of scope

- Cold-cache / out-of-core (store > RAM) regime.
- Sweeping batch size as an independent dimension (fixed by the bp-budget convention).
- Changes to `plot_throughput.py` / `plot_memory.py` semantics.
- Regenerating committed results (a separate run step after implementation).

## Success criteria

- SVAR `elapsed_ns` is unimodal across replicates (no ~10 ms cluster); SVAR
  scatter in `plot.png` collapses to genuine run-to-run variance.
- `n_plot.png` SVAR trend reflects real N-dependence (≈flat), not cluster noise.
- All four formats measured under the identical pair stream and the same
  warmup + sustained-loop protocol.
