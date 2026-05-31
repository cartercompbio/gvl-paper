# Handoff: GVL 0.6.1 memory-benchmark regeneration

_Last updated: 2026-05-30. Author: pairing session with Claude. Cluster going down for
maintenance; this is the resume point._

## TL;DR / where we are

1. **Why we're on 0.6.1:** GVL ≥0.21 (measured on 0.24.1) regressed haplotype/track
   dataloading **~18–20×** in throughput and several-fold in peak RAM vs **0.6.1** — the
   version the manuscript benchmarks were collected on. Confirmed by a controlled parity
   test. Full evidence: `../GenVarLoader/REGRESSIONS.md`. The repo is pinned to 0.6.1 via
   the `bench061` pixi env.
2. **Goal of current work:** regenerate the **avg + peak RSS memory benchmark** on 0.6.1
   (the prior 0.24.1 run OOM-killed at large seqlen and is invalidated anyway).
3. **Done:** ✅ **TCGA_ATAC tracks memory — COMPLETE, all 4 seqlens, full grid, no OOM**
   (`results_gvl061/tracks_memory/`).
4. **Blocked:** haps memory crashes at large batch; 1KGP haps is extremely slow.
   **Most-likely root cause (NOT yet fixed): numba version mismatch** — see next section.

## ⭐ Haps crash — investigated; here's what we know

Two hypotheses were tested:

1. **Numba version (TESTED — pinned, did NOT fix):** `bench061` was floating numba to
   0.65.1; gvl 0.6.1 shipped 2024-11-25 (numba ~0.60). Pinned `numba==0.60.*`
   (llvmlite 0.43.0) — committed in `pixi.toml`. This did **not** fix the crash, but it
   **unmasked the real error** (0.65.1 reported a bare `SystemError`):
   ```
   ValueError: cannot assign slice from input of different size
     in genvarloader/_dataset/__init__.py:989 reconstruct_haplotypes_from_sparse(...)
   ```
2. **Genuine 0.6.1 reconstruction bug at large batch (this is the real cause):** at
   seqlen 16384, **batch 8192 works, batch 16384 crashes** — boundary around
   `n_queries × region_length ≈ 2²⁸`. Tracks never call this kernel, so tracks are fine.

**Why did the original benchmarks run, then?** Still partly open. The original *throughput*
runs (`hap_results.csv`) were 1KGP haps and reached huge batches — so either (a) the bug is
data-dependent (a specific region/sample in the batch triggers the slice-size mismatch, and
the original dataset/region sampling didn't include it), or (b) something about our dataset
build differs. **Worth a focused look on resume** (compare against an original-era dataset,
or bisect which region triggers it). NOTE: batch 8192 is already an absurd training batch —
the crash is only in the unrealistic tail of the sweep.

**RECOMMENDED resume path (pragmatic):** cap the haps memory grid to the stable range and
get clean haps curves; treat the tail as a separate (likely upstream) bug.

```bash
# regenerate a capped haps grid per seqlen (stable: batch <= 8192) and run haps sweeps.
# Easiest: edit make_launch_grid invocation in mem_driver.sh to add a smaller --max-npb
# for the haps pass (e.g. 2**27), OR post-filter the grid CSV to batch_size <= 8192.
# Then: rm results_gvl061/haps_memory/*.csv ; resubmit the array (below).
```
Quick single-cell sanity check (threads=1 on login is fine; batch 8192 OK, 16384 crashes):
```bash
printf "threads,batch_size,n_batches\n1,8192,5\n" > /tmp/tg.csv
NUMBA_NUM_THREADS=1 pixi run -e bench061 python hap_track_throughput/bin_gvl061/benchmark_mem.py \
  /tmp/h.csv data/datasets_gvl061/tcga/seqlen_16384.gvl data/ref/tcga/ref.fa /tmp/tg.csv --mode haps --dataset TCGA_ATAC
```
(bench061 now: numba 0.60.0, numpy 1.26.4, llvmlite 0.43.0.)

## How to resume the memory benchmark

Everything is driven by `mem_driver.sh <tcga|1kgp> <seqlen>` via a throttled SLURM array.

```bash
cd /carter/users/dlaub/projects/gvl-paper
# (after the numba pin above)
sbatch /tmp/mem_array.sh          # 8 tasks, --array=0-7%2 (max 2 concurrent, 150G each)
# array maps: 0-3 = tcga {2048,16384,131072,1048576}; 4-7 = 1kgp {same}
```
`/tmp/mem_array.sh` is ephemeral (node-local /tmp) — **recreate it** (content below) since
the maintenance reboot will wipe /tmp. The driver is **idempotent**: it skips any sweep whose
output CSV already exists, and skips dataset builds that already exist, so re-running only
fills gaps.

### Recreate `/tmp/mem_array.sh`
```bash
cat > /tmp/mem_array.sh <<'SH'
#!/bin/bash
#SBATCH -J memarr
#SBATCH -c 64
#SBATCH --mem 150G
#SBATCH -p carter-compute
#SBATCH -A carter-compute
#SBATCH --array=0-7%2
#SBATCH -o /carter/users/dlaub/projects/gvl-paper/work_gvl061/%x_%A_%a.out
#SBATCH -e /carter/users/dlaub/projects/gvl-paper/work_gvl061/%x_%A_%a.err
COMBOS=(tcga:2048 tcga:16384 tcga:131072 tcga:1048576 1kgp:2048 1kgp:16384 1kgp:131072 1kgp:1048576)
IFS=: read -r DS L <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
bash /carter/users/dlaub/projects/gvl-paper/mem_driver.sh "$DS" "$L"
SH
```

## What is on disk (survives reboot — all under the project on /carter)

| Path | What | Status |
|---|---|---|
| `results_gvl061/tracks_memory/TCGA_ATAC_{2048,16384,131072,1048576}.csv` | TCGA tracks avg+peak RSS | ✅ COMPLETE (full grid) |
| `results_gvl061/haps_memory/TCGA_ATAC_{16384,131072,1048576}.csv` | TCGA haps | ⚠️ PARTIAL (batch ≤ ~8192, crashed beyond). DELETE before re-run so driver redoes them. |
| `results_gvl061/haps_memory/TCGA_ATAC_2048.csv` | TCGA haps 2048 | ⚠️ header-only (crashed at batch 2 @ 64 threads). DELETE. |
| `results_gvl061/haps_memory/1KGP_2048.csv` | 1KGP haps 2048 | ⚠️ PARTIAL. DELETE. |
| `data/datasets_gvl061/tcga/seqlen_*.gvl` | TCGA 0.6.1 datasets (4 seqlens, 61 samples) | ✅ built, reuse |
| `data/datasets_gvl061/1kgp/seqlen_{2048,16384}.gvl` | 1KGP 0.6.1 datasets | ✅ built (2048,16384); 131072/1048576 NOT built |
| `data/tcga_s61/merged.s61.bcf(.csi)` | 61-sample TCGA bcf subset (see quirk #1) | ✅ keep |
| `data/1kgp_stage/hg38.norm.{pgen→sym, pvar(decompressed), psam→sym}` | staged 1KGP variants (quirk #3) | ✅ keep |
| `data/ref/{tcga,1kgp}/ref.fa(+.fai) + ref.fa.gvl/` | staged fastas + pre-built 0.6.1 ref caches (quirk #2) | ✅ keep |
| `parity_ds/` (~265 MB) | throughput parity-test datasets + staged bcf index | disposable (`rm -rf` ok) |

To redo the partial haps CSVs: `rm results_gvl061/haps_memory/*.csv` then resubmit.

## GVL 0.6.1 quirks discovered (already worked around in `mem_driver.sh`)

1. **TCGA bcf sample mismatch:** the bcf has 62 samples, bigwig table 61. gvl 0.6.1's
   parallel genotype reader (`multiprocess_read`) returns inconsistent sample counts
   (61 vs 62) across splits → build crash. Fix: pre-subset bcf to the 61 bigwig samples
   (`data/tcga_s61/merged.s61.bcf`). The dataset is that 61-sample intersection anyway.
2. **Reference cache format clash:** gvl 0.6.1 wants `{fasta}.gvl/` as a **directory**
   cache of per-contig `.npy`. The shared `GRCh38.d1.vd1.fa.gvl` is a stale **3.1 GB file**
   (0.24.1-format) → 0.6.1 sees it "exists", skips rebuild, then `NotADirectoryError`.
   Fix: staged fasta symlinks under `data/ref/{tcga,1kgp}/ref.fa` so 0.6.1 builds its own
   dir-cache; pre-built serially (avoid concurrent-build races across array tasks).
3. **1KGP pvar must be uncompressed:** 0.6.1 `read_pvar` opens a plain `.pvar`; only
   `.pvar.zst` exists. Fix: decompressed to `data/1kgp_stage/hg38.norm.pvar`.

## Remaining issues / open decisions

- **Haps large-batch crash** — try the numba pin FIRST (top of doc). If that fixes it,
  haps should complete the full grid. If not, fallbacks: cap the haps grid to the stable
  range (npb ≤ 2²⁷) and/or run haps at fewer threads (the 64-thread path was where the
  2048 cell crashed at batch 2 — possible numba parallel race).
- **1KGP haps slowness** — even if the crash is fixed, the 1KGP haps *sweep* ran 4+ h/seqlen
  (75 M dense variants × 3202 samples; small-batch grid cells iterate huge `n_batches` =
  `max(10, 2²⁹/npb)`). Likely needs: lower `n_batches` cap in `benchmark_mem.py` / a capped
  batch grid, and/or sample subsampling. **Decision pending with user** (was about to ask:
  reduce 1KGP grid vs drop 1KGP memory vs subsample samples).
- **Was a numba pin used for the throughput parity?** No — the parity test was tracks-only,
  which dodges the crashing kernel. After pinning numba, re-confirm 0.6.1 haps *throughput*
  parity too (1KGP `hap_results.csv`), for completeness.

## Git state

Committed:
- `gvl-paper@baca78d` — `bench061` env (genvarloader==0.6.1) + restored 0.6.1-API scripts.
- `gvl-paper@fcba31f` — `benchmark_mem.py` + `_mem_sampler.py` (avg+peak, per-row flush) + psutil.
- `GenVarLoader@3c067d8, @2ed6a69` — `REGRESSIONS.md` (regression evidence, parity, ruled-out factors).

**Uncommitted (commit on resume):**
- `mem_driver.sh` — the per-(dataset,seqlen) driver (build + grid + sweeps).
- `hap_track_throughput/bin_gvl061/build_ds061.py` — 0.6.1 dataset builder.
- `pixi.toml`/`pixi.lock` will change when numba is pinned.
- `seq_probe.py`, `tbb_probe.py` — ad-hoc probes from the regression investigation (can keep or delete).

`data/`, `work_gvl061/`, `results_gvl061/`? — check `.gitignore` (`data/`, `work/`, `.nextflow*`
are ignored; `results_gvl061/` is NOT ignored — decide whether to commit the memory CSVs).

## Last array job (cancelled at maintenance): 11089510
Tasks 0–4 FAILED (haps crash / partial), 5–6 were the 4 h 1KGP runs (cancelled). TCGA tracks
from an earlier array (11087192) are the complete CSVs now on disk.
