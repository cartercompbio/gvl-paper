#!/bin/bash
#SBATCH --job-name=gvl027-baselines
#SBATCH --partition=carter-compute
#SBATCH --account=carter
#SBATCH --nodelist=carter-cn-03
#SBATCH --exclusive
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl027/baselines_%j.log
#SBATCH --error=/carter/users/dlaub/projects/gvl-paper/results_gvl027/baselines_%j.log
#
# Re-measure FASTA + pyBigWig baselines apples-to-apples with the GVL 0.27 grid.
# Pinned to cn-03 (same node as the 0.27 grid; cn-02 is bandwidth-starved).
# FASTA then pyBigWig, SEQUENTIALLY — two throughput benchmarks on one node would
# contend for DRAM bandwidth and corrupt each other's numbers. Thread counts are
# limited per-cell with taskset so each cell uses exactly T cores (num_workers=T-1).
set -euo pipefail
export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT/hap_track_throughput"

OUT="$ROOT/results_gvl027/baselines"
mkdir -p "$OUT"
THREADS=(1 4 16 32)

for KIND in fasta pybigwig; do
  RES="$OUT/${KIND}.csv"
  rm -f "$RES"   # fresh; the script appends across thread counts
  for T in "${THREADS[@]}"; do
    echo "=== $KIND threads=$T $(date) ==="
    taskset -c 0-$((T-1)) pixi run -e bench027 python bin/benchmark_baseline.py \
      "$KIND" "$T" "$RES"
  done
done
echo "=== DONE $(date) ==="
