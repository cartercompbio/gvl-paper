# Spec: Re-measure FASTA / pyBigWig baselines apples-to-apples with the GVL 0.27 grid

**Date:** 2026-06-17
**Status:** approved design (pending writing-plans)
**Context:** The manuscript is being re-based from GVL 0.6.1 to 0.27 (0.6.1 had correctness
bugs fixed in 0.27). The 0.27 throughput numbers exist (`results_gvl027/{haps,tracks}/*_none.csv`),
but the FASTA (haplotype) and pyBigWig (track) baselines they must be divided by were measured
on the *old* harness, on a different node (cn-04), with a *different* timing loop. To report an
honest speedup we must re-measure those baselines apples-to-apples with the 0.27 grid. See
`text/story.md` and `text/roadmap.md §0`.

## Goal / success criteria
- Produce FASTA and pyBigWig baseline throughput CSVs that are **cell-for-cell comparable**
  to `results_gvl027/{haps,tracks}/*_none.csv`: identical timing loop, byte accounting, schema,
  region tiles, (threads × batch_size × seqlen) grid, and **node (cn-03)**.
- Feed directly into `hap_track_throughput/bin_gvl027/compare_to_baseline.py` to compute
  `speedup = GVL_0.27_max / baseline_max` per seqlen (and per dataset for haps).
- Complete in **< 6 h** wall-clock.
- Baseline is **maximally optimized** (PyTorch multiprocessing DataLoader, `num_workers = n_cpus - 1`)
  so it cannot be dismissed as a strawman — directly answers reviewer R2-min1.

## Non-goals
- No change to the GVL 0.27 numbers or harness.
- Not integrating baselines into `benchmark.nf` (Approach 3, rejected: too much plumbing, slower
  to launch). Standalone script + sbatch instead.
- No GPU / training measurement (that is Fig 2E, separate).

## Design

### Component: `hap_track_throughput/bin/benchmark_baseline.py` (new)
One cyclopts script, two modes via `--kind {fasta,pybigwig}`.

- **Reuses the existing dataset classes** from `benchmark_ref.py` (`Ref`, pysam `FastaFile`) and
  `benchmark_bigwig.py` (`BigWigDataset`, `gvl.BigWigs`). Lift them into this script (or a shared
  `_baseline_datasets.py`) unchanged except as noted.
- **Drives them through `_bench_common.measure_cell`** — the exact timing loop / byte accounting
  the 0.27 GVL numbers use (`burn_in=1, replicates=3, time_limit_s=45, min_batches=5`,
  `THROUGHPUT_HEADER` schema). This is the core of apples-to-apples; the old scripts' bespoke
  loop (`burn_in=5`, replicate-list, separate byte counting) is NOT used.
- **`num_workers = n_cpus - 1`** (full affinity count minus the main process — avoids
  oversubscription; this IS the optimized setting). Emit it / note it for the R2-min1 text.
- **Emits the same schema** with `backend` ∈ {`fasta`, `pybigwig`}, `dl_mode = "none"`:
  `dataset,backend,dl_mode,threads,seqlen,batch_size,n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)`.

### Grid: derived from the 0.27 results (guarantees parity)
Read the distinct `(threads, seqlen, batch_size, n_batches_measured)` tuples directly from:
- FASTA → union over `results_gvl027/haps/*_none.csv` (all datasets; FASTA read cost is
  dataset-independent, matching the original single-FASTA-curve framing).
- pyBigWig → `results_gvl027/tracks/TCGA_ATAC_*_none.csv` (tcga-atac only, matches Fig 2D).

Use each cell's `n_batches_measured` as the `n_batches` target. Seqlens {2048, 16384, 131072,
1048576}; threads {1, 4, 16, 32}; batch sizes as swept by 0.27 (every-other-pow2, up to 8192).

### Memory guard (prevents OOM at large batch × seqlen)
A FASTA batch at batch=8192 × seqlen=1 Mbp is ~8 GB (uint8), exceeding the baseline's RAM
envelope once multiplied by worker prefetch. Guard each cell: if
`batch_size * seqlen * bytes_per_bp * (num_workers+1) * prefetch_factor > mem_cap`, record the
cell as NaN and skip (same NaN convention the 0.27 buffered path used for over-cap cells). Because
the reported statistic is **max throughput per seqlen** and these worker-parallel serial readers do
not peak at the largest batch, dropping over-cap cells does not bias the max. `bytes_per_bp` = 1
(FASTA uint8) / 4 (BigWig float32). `mem_cap` defaults to the 0.27 grid's effective envelope
(parameterize; default ~16 GiB).

### Inputs (verified on disk 2026-06-17)
- **Reference FASTA (fasta mode):** `/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa`
  — GRCh38, on `carter-storage:/carter/users/dlaub`, the **same filesystem** as the SVAR datasets
  (`projectDir/data/datasets/`), so FASTA read I/O is comparable to GVL's. Single baseline curve.
- **BigWig table (pybigwig mode):** `/carter/shared/data/ml4gland/tcga-atac/data/sample_to_bigwig.csv`
  (the same table the tcga-atac GVL tracks were built from).
- **Tile BEDs:** `bin/beds/tile_{length}.bed` **do not currently exist** → regenerate with
  `make_bed.py` so the baseline reads the **same region tiling** the GVL dataset used at each seqlen.
  This is a prerequisite build step.

### Execution: `hap_track_throughput/run_baselines.sbatch` (new)
- `#SBATCH --nodelist=carter-cn-03` (pinned — cn-02 is bandwidth-starved; cn-03 is the node the
  0.27 grid ran on), `--exclusive` (or `--cpus-per-task=32`), `--mem` matching the 0.27 grid's
  per-job request, `--time` headroom (e.g. 8 h).
- Run **FASTA, then pyBigWig, sequentially** — never concurrently on cn-03; two throughput
  benchmarks on one node contend for DRAM bandwidth and corrupt each other's numbers.
- Env: the script reads/writes CSVs and uses pysam/pyBigWig + torch DataLoader. Run under an env
  with those deps. `gvl.BigWigs` is only needed for the bigwig mode; `bench027` provides it. (Confirm
  pysam is available in the chosen env; add if missing.)
- Output → `results_gvl027/baselines/{fasta,pybigwig}.csv`.

### Timing budget (< 6 h)
Per cell ≤ ~45 s (`time_limit_s`), most finish far sooner. ~4 seqlens × 4 threads × ~5 in-cap
batch sizes × 3 replicates ≈ a few hundred cell-runs per kind. Worst case ≈ ~3 h per kind; FASTA
+ pyBigWig sequential ≈ < 6 h. (If tight, the in-cap batch sweep is smaller than the full grid.)

## Analysis / integration
- Extend or invoke `compare_to_baseline.py` to join `results_gvl027/baselines/*` with the GVL
  `*_none.csv` and emit `speedup = GVL_max / baseline_max` per (mode, dataset, seqlen).
- Also re-check the **A100 bandwidth** claim: report the absolute GVL 0.27 MiB/s and compare to
  A100 PCIe (~25 GB/s); per the story decision, soften "exceeds" → "approaches / keeps pace with."

## Error handling
- Empty-epoch / zero-batch cells → `measure_cell` returns None → write NaN row (no infinite loop;
  the 0.26.0-hang guard already lives in `measure_cell`).
- Over-cap cells → NaN (memory guard above).
- Missing tile BED for a seqlen → fail fast with a clear message (regenerate beds first).

## Open items to resolve in the plan
1. Confirm/choose the env that has pysam + pyBigWig + torch + gvl.BigWigs together (likely `bench027`;
   verify pysam present).
2. Confirm `make_bed.py` reproduces the exact region set the GVL datasets were written with (same
   source BED / tiling), so region counts match per seqlen.
3. Confirm the 0.27 grid's per-job `--mem` to mirror it (sets `mem_cap` and sbatch `--mem`).
