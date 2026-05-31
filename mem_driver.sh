#!/bin/bash
# Per (DATASET, SEQLEN): build gvl 0.6.1 dataset (if absent) + run avg/peak RSS memory grid.
# Usage: mem_driver.sh <tcga|1kgp> <seqlen>
set -euo pipefail
DATASET="$1"; L="$2"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT"
BIN=hap_track_throughput/bin_gvl061
GRID_BIN=hap_track_throughput/bin
DSDIR=data/datasets_gvl061/$DATASET
mkdir -p "$DSDIR" results_gvl061/tracks_memory results_gvl061/haps_memory work_gvl061
WORK=work_gvl061/${DATASET}_${L}
mkdir -p "$WORK"
BED=$WORK/tile_${L}.bed
GRID=$WORK/grid_${L}.csv
GVL=$DSDIR/seqlen_${L}.gvl

ROOTD=/carter/users/dlaub/projects/gvl-paper
if [ "$DATASET" = "tcga" ]; then
  # staged fasta symlink -> gvl 0.6.1 builds its own {fasta}.gvl dir-cache here
  # (the shared GRCh38.d1.vd1.fa.gvl is a stale 0.24.1-format *file*, incompatible).
  FASTA=$ROOTD/data/ref/tcga/ref.fa
  # bcf subset to the 61 bigwig samples: gvl 0.6.1's parallel genotype reader returns
  # inconsistent sample counts (62 vs 61) on the full bcf. Dataset is the 61 intersection.
  VARIANTS=$ROOTD/data/tcga_s61/merged.s61.bcf
  TABLE=/carter/shared/data/ml4gland/tcga-atac/data/sample_to_bigwig.csv
  DSNAME=TCGA_ATAC
else
  FASTA=$ROOTD/data/ref/1kgp/ref.fa
  # staged pgen with decompressed .pvar (0.6.1 read_pvar needs uncompressed)
  VARIANTS=$ROOTD/data/1kgp_stage/hg38.norm.pgen
  TABLE=
  DSNAME=1KGP
fi

echo "### node=$(hostname) dataset=$DATASET seqlen=$L cpus=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')"

# 1) BED (genome tiling, ~100 tiles/chrom) -- default env (cyclopts/pyranges1)
pixi run python "$GRID_BIN/make_bed.py" "$L" "$FASTA" "$BED" --canonical --n-samples 100

# 2) memory grid (threads=64, batch sweep to npb=2**33) -- default env
pixi run python "$GRID_BIN/make_launch_grid.py" "$L" --max-npb $((2**33)) --memory-grid --output "$GRID"

# 3) build dataset on gvl 0.6.1 (skip if already built)
if [ -f "$GVL/metadata.json" ]; then
  echo "### dataset exists, skipping build: $GVL"
else
  if [ "$DATASET" = "tcga" ]; then
    pixi run -e bench061 python "$BIN/build_ds061.py" tcga "$GVL" "$BED" "$FASTA" "$L" "$VARIANTS" --bigwig-table "$TABLE" --max-mem-gb 32
  else
    pixi run -e bench061 python "$BIN/build_ds061.py" 1kgp "$GVL" "$BED" "$FASTA" "$L" "$VARIANTS" --max-mem-gb 32
  fi
fi

# 4) memory sweeps on gvl 0.6.1 (skip a sweep if its CSV already exists -> idempotent)
TBBDIR=$ROOT/.pixi/envs/bench061/lib   # ensure numba tbb layer (0.6.1's native)
run_sweep () {  # $1=mode  $2=outdir
  local mode="$1" out="results_gvl061/$2/${DSNAME}_${L}.csv"
  if [ -s "$out" ]; then echo "### skip $mode (exists): $out"; return 0; fi
  NUMBA_NUM_THREADS=64 LD_LIBRARY_PATH="$TBBDIR:${LD_LIBRARY_PATH:-}" \
    pixi run -e bench061 python "$BIN/benchmark_mem.py" \
      "$out" "$GVL" "$FASTA" "$GRID" --mode "$mode" --dataset "$DSNAME" --backend gvl061
}
if [ "$DATASET" = "tcga" ]; then
  run_sweep tracks tracks_memory
fi
run_sweep haps haps_memory

echo "### DONE dataset=$DATASET seqlen=$L"
