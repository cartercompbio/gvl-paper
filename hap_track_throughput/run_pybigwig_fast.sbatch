#!/bin/bash
#SBATCH --job-name=gvl027-pybigwig-fast
#SBATCH --partition=carter-compute
#SBATCH --account=carter
#SBATCH --nodelist=carter-cn-03
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl027/pybigwig_fast_%j.log
#SBATCH --error=/carter/users/dlaub/projects/gvl-paper/results_gvl027/pybigwig_fast_%j.log
#
# Fast re-measure of the pyBigWig (tracks) baseline, same node/scheduling as the
# GVL 0.27 grid (cn-03, cpus=32, NON-exclusive). Replaces the first full run, which
# got stuck for 9.5h on threads=1 giant-batch cells: a single large batch is millions
# of serial pyBigWig reads over NFS and measure_cell can't interrupt mid-batch.
#
# Two changes make this finish in <1h WITHOUT affecting the conclusion:
#   1. --max-batch 16384: skip giant batches. These overhead-bound readers peak at
#      SMALL batches (the completed FASTA baseline peaks at batch<=512, threads 16/32),
#      so giant batches are never the per-seqlen throughput max — only cost. 16384 is a
#      32x margin over FASTA's peak batch. compute_speedups uses only the per-seqlen
#      peak, so capping above the peak region changes nothing it reads.
#   2. THREADS descending (32 16 4 1): the peak lives at high thread counts, so it is
#      measured FIRST — even a timeout still yields the binding number.
# Shorter window (--time-limit-s 10 --replicates 2 --min-batches 2): the peak cells are
# fast; this just bounds the slow low-thread cells. FASTA results (fasta.csv) untouched.
#
# POST-RUN CHECK: confirm the realized per-seqlen peak is NOT at batch==16384 (the cap
# boundary). If it is, the peak may lie above the cap — raise --max-batch and re-run.
set -euo pipefail
export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT/hap_track_throughput"

OUT="$ROOT/results_gvl027/baselines"
mkdir -p "$OUT"
RES="$OUT/pybigwig.csv"
rm -f "$RES"   # fresh; appends across thread counts

for T in 32 16 4 1; do
  echo "=== pybigwig threads=$T $(date) ==="
  pixi run -e bench027 python bin/benchmark_baseline.py pybigwig "$T" "$RES" \
    --max-batch 16384 --replicates 2 --time-limit-s 10 --min-batches 2 --burn-in 1
done
echo "=== DONE $(date) ==="
