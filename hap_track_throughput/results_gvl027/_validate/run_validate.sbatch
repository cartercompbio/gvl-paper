#!/bin/bash
#SBATCH --job-name=gvl027-validate-none
#SBATCH --partition=carter-compute
#SBATCH --account=carter
#SBATCH --nodelist=carter-cn-03
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:25:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl027/_validate/validate_%j.log
set -euo pipefail
export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput
FASTA=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa
pixi run -e bench027 python "$PWD/bin/benchmark_haps.py" \
  results_gvl027/_validate/test_none.csv \
  "$PWD/data/datasets/1kgp/svar/seqlen_2048.gvl" \
  2048 "$FASTA" \
  results_gvl027/_validate/grid_none.csv \
  --dataset 1KGP --backend svar --dl-mode none --replicates 1
echo "VALIDATE_DONE"
