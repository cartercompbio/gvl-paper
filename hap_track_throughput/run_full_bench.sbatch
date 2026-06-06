#!/bin/bash
#SBATCH --job-name=gvl027-full
#SBATCH --partition=carter-compute
#SBATCH --account=carter
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl027/full_%j.log
#SBATCH --error=/carter/users/dlaub/projects/gvl-paper/results_gvl027/full_%j.log
#
# Full GVL 0.27.0 throughput + memory bench across 1kgp, tcga-atac, ukbb.
# This is the Nextflow HEAD job: it stays resident and submits the grid as child
# SLURM jobs (executor=slurm via -profile gvl027). Each pipeline invocation is
# idempotent under -resume; re-submitting resumes from cached work.
#   throughput pass: default to_dataloader + single buffered loader
#   memory pass:     --measure_memory (peak/avg RSS), via -resume reusing SVAR+write
set -euo pipefail

export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT/hap_track_throughput"

# bench_tracks is enabled per-config (tcga-atac.config sets it true); haps always on.
CONFIGS=(configs/1kgp.config configs/tcga-atac.config configs/ukbb.config)

for cfg in "${CONFIGS[@]}"; do
  echo "=== THROUGHPUT $cfg  $(date) ==="
  nextflow run benchmark.nf -profile gvl027 -c "$cfg" -resume

  echo "=== MEMORY $cfg  $(date) ==="
  nextflow run benchmark.nf -profile gvl027 -c "$cfg" --measure_memory -resume
done

echo "=== COMPARE  $(date) ==="
cd "$ROOT"
pixi run -e bench027 python hap_track_throughput/bin_gvl027/compare_to_baseline.py \
  || echo "WARN: compare failed — rerun manually; result CSVs are intact"

echo "=== DONE  $(date) ==="
