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
#   throughput pass: buffered dataloader over a reduced grid (threads {1,4,16,32},
#                    every-other-pow2 batch); all measured jobs pinned to cn-03.
#   best grid:       pick_best_grid.py derives the argmax-throughput (threads,batch)
#                    per (dataset,seqlen) from the throughput CSVs.
#   memory pass:     --measure_memory at only that operating point (no sweep), via
#                    -resume reusing SVAR+write and --best_grid_dir.
set -euo pipefail

export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT/hap_track_throughput"

# bench_tracks is enabled per-config (tcga-atac.config sets it true); haps always on.
CONFIGS=(configs/1kgp.config configs/tcga-atac.config configs/ukbb.config)
RES="$ROOT/results_gvl027"
BEST_GRID="$RES/best_grid"

# 1) Throughput pass for every dataset.
for cfg in "${CONFIGS[@]}"; do
  echo "=== THROUGHPUT $cfg  $(date) ==="
  nextflow run benchmark.nf -profile gvl027 -c "$cfg" -resume
done

# 2) Derive the largest-batch (peak-RAM) operating point per dataset x seqlen.
echo "=== PICK BEST GRID  $(date) ==="
pixi run -e default python bin/pick_best_grid.py \
  "$RES/haps" "$RES/tracks" "$BEST_GRID" --mode largest

# 3) Memory pass: RSS-vs-time growth curve at that point. Only 1kgp + tcga-atac
#    (two datasets are enough to show the mmap growth behavior generalizes;
#    ukbb's huge sample count is not needed and is the heaviest to write).
MEM_CONFIGS=(configs/1kgp.config configs/tcga-atac.config)
for cfg in "${MEM_CONFIGS[@]}"; do
  echo "=== MEMORY $cfg  $(date) ==="
  nextflow run benchmark.nf -profile gvl027 -c "$cfg" \
    --measure_memory --best_grid_dir "$BEST_GRID" -resume
done

echo "=== COMPARE  $(date) ==="
cd "$ROOT"
# compare runs in the default env (it needs matplotlib/seaborn, which bench027
# lacks; it only reads CSVs + plots, no GVL import).
pixi run -e default python hap_track_throughput/bin_gvl027/compare_to_baseline.py \
  || echo "WARN: compare failed — rerun manually; result CSVs are intact"

echo "=== DONE  $(date) ==="
