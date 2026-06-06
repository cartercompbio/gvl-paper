#!/bin/bash
#SBATCH --job-name=gvl026-probe
#SBATCH --partition=carter-compute
#SBATCH --nodelist=carter-cn-04
#SBATCH --account=carter
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl026/sbatch_%j.log
#SBATCH --error=/carter/users/dlaub/projects/gvl-paper/results_gvl026/sbatch_%j.log
#
# Drives the GVL 0.26.0 parity probe end-to-end on carter-cn-04 (the node the
# v0.6.1 baseline was collected on — same microarch keeps the comparison clean).
#   Task 6: build the four 1KGP probe datasets (skipped if already present).
#   Task 7: run probe026.sh over the reduced grid -> results_gvl026/*.csv (idempotent).
#   Task 8: join to baseline -> parity_summary.csv + figures/gvl026_parity.png (non-fatal).
# Re-submitting resumes: built datasets and finished result CSVs are skipped.
set -euo pipefail

export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
# numba's thread-pool ceiling is fixed at import to the cpu count; pin it >= the
# grid's max (64) so set_num_threads(64) is permitted.
export NUMBA_NUM_THREADS=96

ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT"

echo "=== node: $(hostname)  cores: $(nproc)  $(date) ==="

# ---- Task 6: build 1KGP probe datasets ----
FASTA=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa
PGEN=/carter/users/dlaub/data/1kGP/plink2/hg38.norm.pgen
OUTDIR="$ROOT/hap_track_throughput/data/datasets_gvl026/1kgp"
mkdir -p "$OUTDIR"
for L in 2048 16384 131072 1048576; do
  DS="$OUTDIR/seqlen_${L}.gvl"
  if [ -d "$DS" ]; then echo "SKIP build $DS (exists)"; continue; fi
  echo "=== build 1kgp seqlen=$L  $(date) ==="
  pixi run -e bench026 python hap_track_throughput/bin/make_bed.py \
    "$L" "$FASTA" "/tmp/1kgp_${L}.bed" --canonical --n-samples 100
  pixi run -e bench026 python hap_track_throughput/bin/benchmark_write.py \
    "$OUTDIR/write_${L}.csv" "$PGEN" "/tmp/1kgp_${L}.bed" "$FASTA" \
    "$DS" "$L" native --dataset 1kgp --max-mem 16g --max-jitter 0
  echo "BUILT $DS"
done

# ---- Task 7: run the probe ----
echo "=== probe  $(date) ==="
bash "$ROOT/hap_track_throughput/bin_gvl026/probe026.sh"

# ---- Task 8: comparison (non-fatal so a plotting hiccup never discards probe data) ----
echo "=== compare  $(date) ==="
pixi run python "$ROOT/hap_track_throughput/bin_gvl026/compare_to_baseline.py" \
  || echo "WARN: compare_to_baseline.py failed — rerun manually; probe CSVs are intact"

echo "=== DONE  $(date) ==="
