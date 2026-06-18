#!/bin/bash
#SBATCH --job-name=gvl027-base-preflight
#SBATCH --partition=carter-compute
#SBATCH --account=carter
#SBATCH --nodelist=carter-cn-03
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl027/baseline_preflight_%j.log
#SBATCH --error=/carter/users/dlaub/projects/gvl-paper/results_gvl027/baseline_preflight_%j.log
#
# Pre-flight for the FASTA/pyBigWig baseline full run. Same scheduling as the full
# run (cn-03, cpus=32, mem=96G, NON-exclusive). Validates THREE things before
# committing the multi-hour full run:
#   1. No OOM-kill: runs the highest-RSS configs (fasta T=4/16/32, pybigwig T=16/32 —
#      these have the largest modeled resident under cell_fits) under the real 96G cap.
#      Survival => the full run is memory-safe.
#   2. Eviction works end-to-end: fasta runs cold (drop_cache default on); a final
#      contrast cell runs warm (--no-drop-cache) so we can compare throughput.
#   3. Cold-read timing: short per-cell budget (--time-limit-s 15 --replicates 1) gives
#      a wall-time-per-cell figure to re-validate the 8h full-run budget now that
#      FASTA reads are cold (slower than the cached dry-run).
set -euo pipefail
export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT/hap_track_throughput"

FASTA=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa
HAP2=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.hap2.fa
if [ ! -s "$HAP2" ]; then
  echo "Creating hap2 duplicate FASTA (distinct inode to defeat page-cache reuse)..."
  cp "$FASTA" "$HAP2"
  pixi run -e bench027 python -c "import pysam; pysam.faidx('$HAP2')"
fi

OUT="$ROOT/results_gvl027/preflight"
mkdir -p "$OUT"
echo "### cgroup mem limit:"; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true

PFLAGS="--replicates 1 --time-limit-s 15 --min-batches 2 --burn-in 1"

for KT in "fasta 4" "fasta 16" "fasta 32" "pybigwig 16" "pybigwig 32"; do
  set -- $KT; KIND=$1; T=$2
  RES="$OUT/${KIND}_t${T}.csv"; rm -f "$RES"
  echo "=== PREFLIGHT $KIND threads=$T (cold) $(date) ==="
  pixi run -e bench027 python bin/benchmark_baseline.py "$KIND" "$T" "$RES" $PFLAGS
  echo "    peak cgroup mem so far:"; cat /sys/fs/cgroup/memory.peak 2>/dev/null || true
done

# Warm contrast: same fasta T=16 cell set with cache eviction OFF, to confirm
# drop_cache materially lowers throughput (i.e. eviction is actually biting).
echo "=== PREFLIGHT fasta threads=16 (WARM contrast, --no-drop-cache) $(date) ==="
pixi run -e bench027 python bin/benchmark_baseline.py fasta 16 "$OUT/fasta_t16_warm.csv" $PFLAGS --no-drop-cache

echo "=== PREFLIGHT DONE $(date) ==="
echo "### final peak cgroup mem:"; cat /sys/fs/cgroup/memory.peak 2>/dev/null || true
