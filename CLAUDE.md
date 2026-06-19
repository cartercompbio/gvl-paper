# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Benchmarks, training code, and figures for the GenVarLoader (GVL) manuscript. GVL itself lives at https://github.com/mcvickerlab/GenVarLoader; this repo only *consumes* it.

## Environment

Managed with **pixi** (not conda/pip directly). Two environments are defined in `pixi.toml`:

> **The manuscript throughput numbers are now based on GVL 0.27** (`results_gvl027/`), re-based
> from 0.6.1 for correctness (decision 2026-06-17, re-bench completed 2026-06-18). See the
> *GenVarLoader version sensitivity* section below. `scripts/plot.py` reads `results_gvl027/` for
> all four throughput figures; the old `results/{hap,track,ref,pybigwig}_results.csv` (0.6.1) are
> deprecated and kept only for provenance.

- `default` / `bench`: Python 3.12, torch 2.10 (cu126), `genvarloader >=0.24.1,<0.25`. Used for current work and most plotting (`scripts/plot.py` reads CSVs and does **not** import GVL).
- `basenji2`: Python 3.12, torch 2.6 (cu126), `genvarloader ==0.20.0`, `basenji2-pytorch`. Used only for the Basenji2 evaluation notebook/script — pinned to an older GVL on purpose.
- `bench027`: Python 3.12, CPU torch, `genvarloader ==0.27.0`. **The version the manuscript throughput + memory numbers are now collected on** (`results_gvl027/`), via the production Nextflow harness `hap_track_throughput/benchmark.nf -profile gvl027`. See the version-sensitivity section for the full protocol.
- `bench061`: Python 3.12, CPU torch, `genvarloader ==0.6.1`. The version the *original* (now superseded) manuscript throughput benchmarks (`results/{hap,track}_results.csv`) were collected on. GVL ≥0.21 regressed haplotype/track dataloading ~10–30× in throughput and several-fold in peak RAM (see `../GenVarLoader/REGRESSIONS.md`); the paper accepts 0.27's lower throughput as the cost of correctness. Kept for provenance with the 0.6.1-API scripts in `hap_track_throughput/bin_gvl061/`. Do **not** use these CSVs for the manuscript anymore.

Run anything in an env with `pixi run -e <env> <cmd>` (default env is implicit). Example: `pixi r scripts/plot.py`. Register Jupyter kernels via the `i-kernel` task in each feature.

`env.yaml` is a stale conda recipe kept for reference — use pixi.

## Repo layout (big picture)

- `throughput/` — the main benchmark harness.
  - `benchmark.nf` is a Nextflow DSL2 pipeline that builds GVL datasets at several sequence lengths and runs haplotype / track benchmarks across a grid of (threads, batch_size). Datasets are configured via `configs/{1kgp,gdc,tcga-atac,ukbb}.config`.
  - `bin/` contains the scripts the pipeline invokes (`make_bed.py`, `benchmark_haps.py`, `benchmark_tracks.py`, `benchmark_ref.py`, `benchmark_bigwig.py`, `make_launch_grid.py`). Nextflow places `bin/` on PATH automatically.
  - `launch_benchmarks.py` / `launch_ref_benchmarks.py` / `launch_bigwig_benchmarks.py` drive SLURM submissions outside Nextflow; they hardcode partitions/nodes and must be edited for other clusters.
  - `bench_gvl_vcf_plink.py` is the variant-throughput comparison against raw VCF/PLINK.
  - Results land in `results/` and are consumed by notebooks + `scripts/plot.py`.
- `gpu_utilization/` — BPNet training run used to measure GPU utilization. `train_BPNet.py` ties together `arch.BPNetHaps`, `dataloader.ATACDataModule`, and `metrics.{bpnetlite_loss,bpnetlite_metrics}` via PyTorch Lightning + WandB. **Note the warning at the top of `train_BPNet.py`**: BPNet metrics require removing the `.squeeze()` calls from `seqmodels.Module` — this is an unpatched upstream issue.
- `basenji2/` — Basenji2 reproduction. Inference happens in `basenji2-eval-hg19.ipynb` (uses the `basenji2` pixi env); predictions are cached at `basenji2/data/preds_hg19.npy` and turned into figures by `scripts/plot_basenji2.py`.
- `borzoi/` — Borzoi evaluation notebook (`borzoi-eval.ipynb`).
- `notebooks/` — throughput analysis (`hap_and_track_throughput.ipynb`, `variant_throughput.ipynb`).
- `scripts/plot.py` is the canonical figure-generation entry point for everything except Basenji2.
- `figures/`, `results/`, `data/` — outputs/inputs, mostly gitignored or controlled-access.

## Common commands

```bash
# Regenerate all figures + archive code (top-level reproducibility script).
# Expects RNA-seq data at /carter/users/dlaub/data/1kGP-rna-seq — edit the path in run_scripts.sh on other machines.
bash run_scripts.sh

# Single figure pass (no Basenji2, no archive):
pixi r scripts/plot.py

# Basenji2 figures only (needs the basenji2 env's deps but the script runs under default):
pixi r scripts/plot_basenji2.py basenji2/gene_list.csv ... basenji2/data/preds_hg19.npy

# Run the throughput pipeline for a dataset (edit/select a config first):
cd throughput && nextflow run benchmark.nf -c configs/1kgp.config

# BPNet training (single run):
pixi r python gpu_utilization/train_BPNet.py
```

## 1kGP benchmark reproduction

Per `README.md`: download the Zenodo tarballs into `throughput/datasets/1kgp/`, drop the GRCh38 1000G reference FASTA into `throughput/`, and edit SLURM specifics in `throughput/launch_benchmarks.py` (queue names, node availability) before running.

## GenVarLoader version sensitivity

GVL's API and performance both shifted across releases, so the env is split four ways:

**Source of truth: the manuscript throughput + memory numbers are GVL 0.27 (`results_gvl027/`).**
0.6.1 (the original benchmark version) had correctness bugs fixed in 0.27, so the paper was
re-based entirely onto 0.27 (decision 2026-06-17). 0.27 is slower than 0.6.1 (best-operating-point
throughput ≈ 30–55% of 0.6.1), so headline numbers dropped, and the paper accepts this as the cost
of correctness. The FASTA/pyBigWig baselines were re-measured on the **same hardware** (cn-03) as
the 0.27 grid, so speedups are apples-to-apples: `results_gvl027/baselines/{fasta,pybigwig}.csv`,
ratios in `results_gvl027/speedups.csv` (haplotypes 7.8–16.8× vs FASTA, tracks 9.4–128.9× vs
pyBigWig; no cell reaches A100 PCIe bandwidth).

- `feature.basenji2` → `genvarloader ==0.20.0`: don't bump it — the cached `preds_hg19.npy` and `basenji2-eval-hg19.ipynb` target the 0.20 API.
- `feature.bench061` → `genvarloader ==0.6.1`: the version the *original* manuscript throughput results were measured on, **now superseded by 0.27**. **GVL ≥0.21 regressed dataloading throughput ~10–30× and peak RAM several-fold** (root cause not yet found upstream; documented in `../GenVarLoader/REGRESSIONS.md`). `results/{hap,track}_results.csv` and the `hap_track_throughput/bin_gvl061/` scripts are kept for provenance only; do **not** use them for the manuscript.
- `feature.bench` (default) → `genvarloader >=0.24.1`: current/general work and plotting (`scripts/plot.py`, which reads CSVs and does not import GVL). **Not** for regenerating benchmark numbers.
- `feature.bench027` → `genvarloader ==0.27.0`: **the manuscript throughput + memory source of truth** (bumped from 0.26.0 — see below). The full grid is run through the production Nextflow harness `hap_track_throughput/benchmark.nf -profile gvl027`; outputs land in `results_gvl027/` and feed all four throughput figures in `scripts/plot.py`. (It began as a parity probe vs the 0.6.1 baseline in `hap_track_throughput/bin_gvl027/`; that comparison is now an internal sanity check.) CPU torch (no GPU workload).
  **Why 0.27.0:** 0.26.0's `buffered` dataloader forced `drop_last=True`, so a cell with
  `batch_size > n_instances` yielded 0 batches — which sent `benchmark_dl.py`'s `while not done`
  loop into an infinite re-iteration (hung the tcga-atac probe cells, which have only 6200
  instances; 1kGP with ~7.7M was unaffected). Fixed in 0.27.0.
  **Full throughput+memory bench (2026-06-05):** the same `bench027` env now also
  backs the *full* manuscript grid (not just the reduced probe), run through the
  production Nextflow harness `hap_track_throughput/benchmark.nf` with
  `-profile gvl027` (see `hap_track_throughput/nextflow.config`, which activates
  this env on each SLURM job). All GVL datasets are **SVAR-backed** (`bench_native`
  off by default); genoray 2.9.0's hap-safe filter (`~is_symbolic & ~is_breakend`,
  via `bin/_genoray_filter.py`) is applied at SVAR conversion. Each cell is run in
  the **eager `mode=none` dataloader only** (plain torch `DataLoader` over
  `to_torch_dataset`), so every measured mini-batch is a real per-batch decode.
  This matches the 0.6.1 baseline exactly (0.6.1's `to_dataloader` had **no**
  buffered path) and is artifact-free. **The `buffered` mode was tried first and
  abandoned (2026-06-08):** `mode='buffered'` (`n_slots=1`, single-slot
  super-batch — *not* the async `double_buffered`) decodes a whole chunk once via
  `_buffered_loader.py` then yields cheap array slices, so a short measurement
  window (the grid stops at `n_batches=10` for large batches) clocks
  slice-handoff, reporting >1 TB/s — physically impossible above cn-03's ~35 GB/s
  STREAM-Triad DRAM bandwidth, i.e. the bytes were never streamed. Eager mode has
  no chunk to slice, so the artifact cannot occur. The grid still sweeps
  batch_size (threads {1,4,16,32} × every-other-pow2 batch); `max_buffer_bytes`
  and the per-cell buffer-sizing in `benchmark_{haps,tracks}.py` are dead code
  under `mode=none` (no buffer is allocated, so the large-batch NaN problem also
  disappears).
  `bin/benchmark_haps.py` / `benchmark_tracks.py` emit the
  reconciled schema `dataset,backend,dl_mode,threads,seqlen,batch_size,
  n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)` (memory pass:
  `...,avg_rss_bytes,peak_rss_bytes`), directly comparable to the 0.6.1 baseline.
  Drive the whole thing with `hap_track_throughput/run_full_bench.sbatch`; outputs
  land in `results_gvl027/{haps,tracks,haps_memory,tracks_memory,...}` and are joined
  to the baseline by `bin_gvl027/compare_to_baseline.py`. The throughput-vs-0.6.1
  ratio is an **internal sanity check**; memory is reported as 0.27.0 absolute (no
  0.6.1 memory baseline exists). The reported speedups instead come from the
  same-hardware FASTA/pyBigWig re-bench in `results_gvl027/baselines/` (see the
  source-of-truth note above and `results_gvl027/speedups.csv`).
