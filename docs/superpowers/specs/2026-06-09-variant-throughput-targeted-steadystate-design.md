# Variant-throughput: targeted steady-state (speed fix)

**Date:** 2026-06-09
**Scope:** `variant_throughput/` benchmark harness
**Status:** design approved, pending implementation plan
**Supersedes (partially):** `2026-06-09-variant-throughput-steady-state-design.md` —
keeps that spec's SVAR steady-state fix; revises the I/O formats and stream size.

## Problem

The steady-state redesign (prior spec, now implemented on
`feat/gvl-027-throughput-bench`) made the benchmark ~400× slower per cell. A full
`1kgp.config` run sat for hours on the largest cells (`q65536_n3202`,
`q262144_n3202`) and was killed.

Two compounding causes, both concentrated on the **I/O formats** (BCF, PGEN,
PRESUB-BCF), which have **no** thread-pool parking artifact (the artifact the
redesign targeted is numba-specific, i.e. SVAR-only — the prior spec itself notes
the other three "scatter tightly"):

1. **Untimed multi-pass overhead.** `run_stream` runs, per replicate: one full
   pass to count `distinct_calls`, then `prime(passes=2)` = two more full passes,
   then a sustained `drive_loop` (`min_batches=10` AND `min_seconds=5`). For I/O
   formats a single batch read is *seconds*, so the three untimed passes over
   `stream_batches=64` batches dominate (~16 min/replicate of warmup). The
   implementation also diverged from the prior spec, which specified warmup "over
   a few batches," not two full passes.
2. **Stream size blew up 64×.** Old `generate_pairs` emitted a random `1..100`
   pairs per replicate capped at `2²⁴` bp ≈ **½ a batch**. The redesign emits
   `stream_batches=64 × batch_size` = **64 full batches**. Even a pure single-pass
   revert over this stream would be ~128× the old per-replicate work.

Additionally, `min_batches=10` is a trap for slow formats: once one batch exceeds
`min_seconds`, the loop must still run 10. `min_batches` only ever *binds* for slow
(I/O) formats — never for sub-ms SVAR, where `min_seconds` already yields millions
of iterations — i.e. backwards from its intent.

## Decision

Apply the steady-state machinery **only to SVAR**, where the artifact exists.
Revert the three I/O formats to their old single-timed-pass behavior. Shrink the
stream so per-replicate cost returns to roughly the old magnitude.

This relaxes the prior spec's success criterion #3 ("all four formats under the
same warmup + sustained-loop protocol"). That is intentional and justified: the
I/O formats have no parking artifact, scattered tightly under single-pass timing,
and are the slow ones. Each format is still measured at its true steady-state
cost — SVAR via a hot loop (removing the artifact), the I/O formats via a direct
pass (which has no artifact to remove).

## Design

### 1. SVAR — unchanged
`bench_svar.py` keeps: AOT `_svar_search` per batch (timed once → `setup_ns`),
cached indices, sustained `_svar_pack` replay via `run_stream`
(`min_seconds`/`min_batches`). Both throughput and memory modes unchanged. This
keeps the prior spec's primary success criterion: SVAR `elapsed_ns` unimodal, no
~10 ms cluster.

### 2. BCF / PGEN — single timed pass per replicate
Per replicate: split the stream into batches (shared `split_pair_batches`), run
**one** timed pass over all batches — no warmup, no sustained loop. Report:
- `n_calls` = total nonzero genotypes over the pass
- `elapsed_ns` = wall time of the pass
- `setup_ns` = `None`
- `n_pairs` = pairs in the pass (`stream_batches × batch_size`)

Memory mode: the same single pass under `PeakRssSampler` (old behavior).
Drop the `min_seconds` / `min_batches` parameters from these two scripts.

### 3. PRESUB-BCF — AOT subset + single timed read pass
Keep the AOT `bcftools` subset of each batch into temp BCFs (timed once →
`setup_ns`) and the temp-BCF cleanup in `finally`. Replace the sustained replay
with a **single** timed `cyvcf2` read pass over the cached temp BCFs (throughput),
or one pass under `PeakRssSampler` (memory). Drop `min_seconds` / `min_batches`.

### 4. Stream size
`stream_batches` default **64 → 8**. Because `batch_size = bp_budget //
query_length`, every batch reads ~constant total bytes (~16 MiB), so per-replicate
cost is ~constant across query lengths and scales ~linearly with `stream_batches`.
8 distinct batches give SVAR's 5 s hot loop plenty to cycle and the I/O formats a
representative single pass (≈8× the old per-replicate workload), while keeping the
n3202 BCF/PGEN cells in the seconds range. `generate_pairs.py` is otherwise
unchanged (it already emits the fixed batched stream with `batch_id`).

### 5. Nextflow wiring
- `params.stream_batches` default → 8.
- `min_seconds` / `min_batches` flags remain on `BENCH_SVAR_THROUGHPUT` and
  `BENCH_SVAR_MEMORY` only; remove them from the six I/O `BENCH_*` script blocks.
- `GENERATE_PAIRS` / `GENERATE_PAIRS_N` unchanged.

### 6. CSV schema + plotting — unchanged
Throughput: `dataset, method, query_length, n_samples, replicate, n_pairs,
n_calls, elapsed_ns, setup_ns`. Memory: `..., n_calls, peak_rss_bytes`. The
per-row rate `n_calls / elapsed_ns` stays valid even though I/O `n_pairs` may
differ from SVAR's distinct-stream count (each row's rate is self-consistent).
`plot_throughput.py` / `plot_memory.py` / `_plot_common.py` untouched.

## Out of scope
- Regenerating committed manuscript results (separate run step after the fix).
- Changing the query-length × sample-size grid.
- Any further change to `bench_svar.py` or `_streaming.py` internals.

## Success criteria
- A full `1kgp.config` run completes in a tractable time (low hours, not days);
  no cell hangs for hours.
- SVAR `elapsed_ns` remains unimodal across replicates (artifact still gone).
- BCF / PGEN / PRESUB-BCF throughput rows are produced by a single timed pass and
  scatter comparably to the pre-redesign results.
- CSV schema and plotting outputs unchanged; existing tests pass.
