# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Benchmarks, training code, and figures for the GenVarLoader (GVL) manuscript. GVL itself lives at https://github.com/mcvickerlab/GenVarLoader; this repo only *consumes* it.

## Environment

Managed with **pixi** (not conda/pip directly). Two environments are defined in `pixi.toml`:

- `default` / `bench`: Python 3.12, torch 2.10 (cu126), `genvarloader >=0.24.1,<0.25`. Used for current work and most plotting (`scripts/plot.py` reads CSVs and does **not** import GVL).
- `basenji2`: Python 3.12, torch 2.6 (cu126), `genvarloader ==0.20.0`, `basenji2-pytorch`. Used only for the Basenji2 evaluation notebook/script — pinned to an older GVL on purpose.
- `bench061`: Python 3.12, CPU torch, `genvarloader ==0.6.1`. The version the manuscript throughput benchmarks (`results/{hap,track}_results.csv`) were collected on. GVL ≥0.21 regressed haplotype/track dataloading ~10–30× in throughput and several-fold in peak RAM (see `../GenVarLoader/REGRESSIONS.md`). Re-run paper throughput benchmarks here with the 0.6.1-API scripts in `hap_track_throughput/bin_gvl061/`; do **not** regenerate those CSVs on a newer GVL.

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

GVL's API and performance both shifted across releases, so the env is split three ways:

- `feature.basenji2` → `genvarloader ==0.20.0`: don't bump it — the cached `preds_hg19.npy` and `basenji2-eval-hg19.ipynb` target the 0.20 API.
- `feature.bench061` → `genvarloader ==0.6.1`: the version the manuscript throughput results were measured on. **GVL ≥0.21 regressed dataloading throughput ~10–30× and peak RAM several-fold** (root cause not yet found upstream; documented in `../GenVarLoader/REGRESSIONS.md`). Regenerate `results/{hap,track}_results.csv` only in this env, using `hap_track_throughput/bin_gvl061/`.
- `feature.bench` (default) → `genvarloader >=0.24.1`: current/general work and plotting. **Not** for regenerating the paper throughput numbers (it would silently ship the regressed values).
