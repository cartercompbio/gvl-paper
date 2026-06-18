#!/bin/bash
#SBATCH --job-name=gvl027-baselines
#SBATCH --partition=carter-compute
#SBATCH --account=carter
#SBATCH --nodelist=carter-cn-03
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl027/baselines_%j.log
#SBATCH --error=/carter/users/dlaub/projects/gvl-paper/results_gvl027/baselines_%j.log
#
# Re-measure FASTA + pyBigWig baselines apples-to-apples with the GVL 0.27 grid.
# Scheduling MATCHES how the 0.27 GVL numbers were collected (benchmark.nf:205,243):
# --nodelist=carter-cn-03 --cpus-per-task=32 --mem=96G, NON-exclusive (shared node,
# same co-tenant contention regime as GVL; cn-02 is bandwidth-starved so cn-03 only).
# FASTA then pyBigWig, SEQUENTIALLY. Thread counts are swept by num_workers=T-1 (one
# invocation per T) — count-based parallelism like GVL's nb.set_num_threads(T); NOT
# taskset core-pinning (fragile under a shared cgroup that hands out non-contiguous
# core IDs, and not how GVL swept threads).
#
# Dry-run timing (cn-03, job 11611503, 2026-06-17):
#   kind=fasta, threads=1, 4 seqlens, time_limit_s=10, replicates=1, burn_in=1
#   → 17m 28s wall, 38 cells, avg 27.6s/cell, peak RSS 27 GiB.
#   Full-run (time_limit_s=45, replicates=3) upper-bound estimate:
#     fasta: 138 cells; pybigwig: 124 cells; total 262 cells.
#     Upper-bound avg = 27.6s × (45/10) × 3 = 373s/cell × 262 = 97,826s (~27h).
#     Realistic (mix of fast/slow cells; higher T faster; pybigwig shallower BED):
#     ~5-6h. Fits under 8h limit comfortably. No mitigation needed.
# Note: fasta baseline is now 2 bytes/bp (diploid — reads two haplotype FASTA files
# per item), so fewer large-batch fasta cells fit within the 96 GiB RAM cap.
# Note: fasta baseline runs with drop_cache=True (default on) — posix_fadvise DONTNEED
# evicts each hap file's pages after every read, so every fetch is a cold storage read,
# mirroring a real bcftools-consensus cohort where 2*n_samples FASTAs never fit in RAM.
set -euo pipefail
export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT/hap_track_throughput"

# Provision the hap2 duplicate FASTA (distinct inode defeats OS page-cache reuse).
FASTA=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa
HAP2=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.hap2.fa
if [ ! -s "$HAP2" ]; then
  echo "Creating hap2 duplicate FASTA (distinct inode to defeat page-cache reuse)..."
  cp "$FASTA" "$HAP2"
  pixi run -e bench027 python -c "import pysam; pysam.faidx('$HAP2')"
fi

OUT="$ROOT/results_gvl027/baselines"
mkdir -p "$OUT"
THREADS=(1 4 16 32)

for KIND in fasta pybigwig; do
  RES="$OUT/${KIND}.csv"
  rm -f "$RES"   # fresh; the script appends across thread counts
  for T in "${THREADS[@]}"; do
    echo "=== $KIND threads=$T $(date) ==="
    # threads swept via num_workers=T-1 (count-based, like GVL); no taskset core-pin.
    pixi run -e bench027 python bin/benchmark_baseline.py "$KIND" "$T" "$RES"
  done
done
echo "=== DONE $(date) ==="
