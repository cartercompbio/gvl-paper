# GVL 0.27.0 reduced-grid full bench (cn-03, buffered-only) — design

**Date:** 2026-06-06
**Branch:** feat/gvl-026-parity-probe

> **CORRECTION (2026-06-08): switched from `buffered` to eager `mode=none`.**
> The "amortization artifact" noted below for the memory pass turned out to
> contaminate the *throughput* numbers too, fatally. `mode='buffered'`
> (`n_slots=1`, single-slot super-batch in GVL's `_buffered_loader.py` — *not*
> the async `double_buffered`) decodes a whole chunk once, then yields cheap
> array slices. With the grid's short measurement window (`n_batches=10` at large
> batch), the timer clocks slice-handoff, not real dataloading — a 1KGP haps cell
> reported **4,355 GB/s** in a 0.31 ms window, physically impossible above
> cn-03's ~35 GB/s STREAM-Triad DRAM bandwidth. Eager `mode=none` (plain torch
> `DataLoader`, real per-batch decode) has no chunk to slice, so the artifact
> cannot occur, and it matches the 0.6.1 baseline exactly (0.6.1 had no buffered
> path). The harness now runs `dl_modes = ["none"]`; the per-cell buffer-sizing /
> `max_buffer_bytes` machinery is dead code under `mode=none` (and the large-batch
> NaN problem disappears with it). Buffered full-bench outputs were archived to
> `results_gvl027/_buffered_archive/`. Everything below describing "buffered"
> applies historically only.

## Goal

Run the full GVL 0.27.0 throughput + memory benchmark via the Nextflow harness
(`hap_track_throughput/benchmark.nf`, `-profile gvl027`), restricted to a
**reduced grid** that still supports the same conclusions as the full GVL 0.6.1
benchmark, with memory measured only at the **best-throughput operating point**.
Buffered dataloader only. Threads capped at 32. All measured jobs pinned to a
single node.

## Node decision: carter-cn-03 only

`carter-cn-02` and `carter-cn-03` were tested for hardware parity (the property
that matters for benchmark comparability):

- **CPU microarch — identical.** Both Intel Xeon E5-4650 v3 @2.10GHz, family
  6/model 63/stepping 2, 4×12×2 = 96 threads, identical caches and CPU-flag set
  (sha `1d7fdbba246a0e1b`). Only CPU microcode rev differs (0x38 vs 0x2e;
  negligible).
- **Disk — identical.** Both reach the *same* `carter-storage` NFS filer over
  identical 10GbE `ixgbe` links (`TmpDisk=0`, no local scratch). dd O_DIRECT:
  ~400 MB/s write, ~500 MB/s read on both (within noise).
- **RAM bandwidth — NOT the same.** STREAM-Triad:
  - single-thread local: 10.77 (cn-02) vs 10.92 (cn-03) GB/s → same DIMM
    speed/model.
  - 12-thread local: **12.35 (cn-02) vs 38.45 (cn-03) GB/s**; 32-thread
    interleaved: **14.0 vs 35.1 GB/s**, stable to 3 sig-figs over 3 reps.
  - cn-02 barely scales past one core (memory-channel-starved: few populated
    channels/socket), so its multithreaded memory bandwidth is ~40% of cn-03.

The SLURM `RealMemory` gap (953674 vs 476837 MB) is a scheduler-config
difference, not physical — both nodes physically have ~1 TB (4×258 GB NUMA).

Per the rule "if not the same hardware, use cn-03," and because cn-03 is also the
higher-bandwidth, currently-cleaner node, **the entire benchmark is pinned to
carter-cn-03.**

## Conclusions the 0.6.1 grid supports (must be preserved)

From `scripts/plot.py` over `results/{hap,track}_results.csv`:

1. `hap_throughput`: throughput (GB/s) vs nucleotides-per-batch (seqlen×batch),
   one line per thread count, GVL vs FASTA, with the 31.5 GB/s A100 transfer
   limit. Needs threads sweep × batch sweep.
2. `track_throughput`: same shape, GVL vs pyBigWig (tcga-atac only).
3. Peak hap throughput per (dataset, seqlen) + ratio vs FASTA (`max` over grid).
4. Peak track throughput per (dataset, seqlen) + ratio vs BigWig.

(1)–(2) need the *shape* (a few thread lines + a batch spread); (3)–(4) need only
the per-(dataset,seqlen) *peak*.

## Reduced throughput grid

- **Seqlens:** all 4 — `{2048, 16384, 131072, 1048576}`.
- **Threads:** `{1, 4, 16, 32}` (was `{1,2,4,8,16,32,64}`). 4 lines still show
  scaling/saturation; 64 excluded by the cap.
- **Batch sizes:** every *other* power of 2 across each seqlen's valid
  `[min_bs, max_bs]`, with `max_bs` always included (peak region). ~6–8
  points/seqlen vs 14–23.
- ≈ 1/3 the cells; same four figures qualitatively.

## Memory: RSS growth at the largest-batch operating point (no sweep)

Revised after observing that buffered throughput at small/moderate batches is an
**amortization artifact** (the 2 GiB buffer holds many minibatches, so the short
measurement window never drains it — throughput reads in the millions of MiB/s).
The argmax-throughput cell therefore lands on tiny batches whose RSS is just the
floor buffer — uninformative. Instead:

- `bin/pick_best_grid.py --mode largest` picks, per `(dataset, seqlen)`, the
  **largest batch_size among valid (non-NaN) throughput cells** (= largest batch
  within the 64 GiB buffer cap), at the highest thread count. Peak RSS is
  monotonic in batch size, so this is the peak-RAM operating point.
- The memory pass then captures an **RSS-vs-time growth curve** at that point:
  a background sampler records `(elapsed_ns, rss_bytes)` at ~2 Hz, flushed
  incrementally, while the dataloader is iterated for a fixed window
  (`growth_time_s`, default 180 s). Hypothesis: mmap-backed RSS climbs as pages
  fault in and does not settle at a small working set; within the SLURM `--mem`
  cgroup it plateaus near the cap (kernel reclaims clean file-backed pages) —
  the curve demonstrates the unbounded working set without swapping the shared
  node. Peak/avg are derivable from the series.
- Run for **1kgp + tcga-atac only** (haps for both; tracks for tcga-atac) — two
  datasets are enough to show the behavior generalizes; ukbb is skipped.
- Output schema: `dataset,backend,dl_mode,threads,seqlen,batch_size,elapsed_ns,
  rss_bytes` in `results_gvl027/{haps_memory,tracks_memory}/`.

## Implementation

1. `bin/make_launch_grid.py`: thread set → explicit `{1,4,16,32}`; batch loop →
   `range(min_bs, max_bs+1, 2)` plus forced `max_bs`. Drop the `--memory-grid`
   batch-sweep path (memory now uses the derived best grid).
2. `bin/pick_best_grid.py` (new): derive best-setting one-row grids from the
   throughput CSVs (haps + tracks separately).
3. `benchmark.nf`: `BENCH_HAPS`/`BENCH_TRACKS` — `--nodelist=carter-cn-03`;
   `cpus … : 64` → `: 32`. New `params.best_grid_dir` (default null): when
   `measure_memory && best_grid_dir`, skip `make_launch_grid.py` and pass
   `${best_grid_dir}/${dataset}_${length}_{haps,tracks}.csv` as the grid.
4. `nextflow.config`: `NUMBA_NUM_THREADS=64` → `32`.
5. `run_full_bench.sbatch`: per config, run throughput pass → `pick_best_grid.py`
   → memory pass with `--best_grid_dir`. Then `compare_to_baseline.py`.
6. Buffered-only (`dl_modes = ["buffered"]`) — already in place.

## Out of scope / non-goals

- No GPU workload (CPU torch in bench027).
- Do not repoint manuscript baseline CSVs to 0.27.0; throughput-vs-0.6.1 ratio
  is an internal sanity check, memory is reported as 0.27.0 absolute.
