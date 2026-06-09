#!/bin/bash
# Idempotent driver for the GVL 0.27.0 parity probe.
# Runs (dataset,output-mode) x dl-mode x seqlen over the reduced grid, writing
# one CSV per combo into results_gvl027/. Skips combos whose CSV already exists.
set -euo pipefail

ROOT=/carter/users/dlaub/projects/gvl-paper
BIN="$ROOT/hap_track_throughput/bin_gvl027"
OUT="$ROOT/results_gvl027"
mkdir -p "$OUT"

TCGA_FASTA=/cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa
KGP_FASTA=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa
TCGA_DS_DIR="$ROOT/hap_track_throughput/data/datasets/tcga-atac"
KGP_DS_DIR="$ROOT/hap_track_throughput/data/datasets_gvl027/1kgp"

SEQLENS=(2048 16384 131072 1048576)
DLMODES=(none buffered)

# combo := "dataset:output_mode:ds_dir:fasta"
COMBOS=(
  "1kgp:haps:$KGP_DS_DIR:$KGP_FASTA"
  "tcga-atac:haps:$TCGA_DS_DIR:$TCGA_FASTA"
  "tcga-atac:tracks:$TCGA_DS_DIR:$TCGA_FASTA"
)

cd "$BIN"
for combo in "${COMBOS[@]}"; do
  IFS=: read -r dataset omode ds_dir fasta <<< "$combo"
  for L in "${SEQLENS[@]}"; do
    grid="/tmp/probe_grid_${L}.csv"
    [ -f "$grid" ] || pixi run -e bench027 python make_probe_grid.py "$L" --output "$grid"
    ds="$ds_dir/seqlen_${L}.gvl"
    if [ ! -d "$ds" ]; then echo "MISSING dataset $ds — skipping"; continue; fi
    for dl in "${DLMODES[@]}"; do
      res="$OUT/${dataset}_${omode}_${L}_${dl}.csv"
      if [ -f "$res" ]; then echo "SKIP $res (exists)"; continue; fi
      echo "RUN $dataset $omode seqlen=$L dl=$dl"
      tmp="${res}.partial"
      pixi run -e bench027 python benchmark_dl.py "$tmp" "$ds" "$L" "$fasta" "$grid" \
        --mode "$omode" --dl-mode "$dl" --dataset "$dataset"
      mv "$tmp" "$res"   # atomic: final CSV appears only on clean completion
    done
  done
done
echo "PROBE COMPLETE -> $OUT"
