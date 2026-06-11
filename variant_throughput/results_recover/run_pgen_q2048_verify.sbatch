#!/bin/bash
#SBATCH -p carter-compute
#SBATCH --nodelist=carter-cn-04
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J pgen_q2048_verify
#SBATCH -o /carter/users/dlaub/projects/gvl-paper/variant_throughput/results_recover/pgen_q2048_verify.log
cd /carter/users/dlaub/projects/gvl-paper/variant_throughput
pixi run python bin/bench_pgen.py \
  work/e9/ed656b70824e49e819f1358f476563/pairs_2048.parquet \
  /carter/users/dlaub/data/1kGP/1kGP.snp_indel.split_multiallelics.pgen \
  results_recover/pgen_q2048_verify4096_memory.csv \
  --dataset 1KGP --mode memory --n-samples 3202 --max-pairs-per-rep 4096
