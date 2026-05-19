# Variant Random-Access Throughput Benchmark — Methods

## Overview

We benchmarked the throughput of random genotype lookups across four file formats: GVL's SparseVar (SVAR), BCF, PGEN, and a pre-subsetting BCF workflow (PRESUB-BCF). The primary metric is the number of non-reference allele calls retrieved per second as a function of genomic query length. For methods with a distinct setup phase (SVAR index search; PRESUB-BCF sample subsetting), we additionally report setup throughput and peak memory separately from read throughput.

## Dataset

Benchmarks were run against the 1000 Genomes Project (1KGP) whole-genome variant callset (SNPs and indels, multi-allelic sites split), stored in each of the four formats from the same underlying data. The reference genome was GRCh38 with full analysis set plus decoy and HLA contigs.

## Query generation

For each query length $L \in \{2048, 4096, 8192, \ldots, 16{,}777{,}216\}$ bp (14 values, powers of 2), a set of (genomic region, sample) pairs was drawn uniformly at random. Sampling was stratified to exclude assembly gap regions (UCSC hg38 gap table) and restricted to contigs present in all four format files. Within each eligible contig, start positions were drawn uniformly from all positions that keep the query fully within a non-gap interval, weighted proportionally to the available span. For each query length, up to `max_pairs = 100` pairs were sampled per replicate subject to a total-bases cap of 16 Mb (`max_total_length`), yielding between 1 and 100 pairs depending on query length. Five independent replicates were generated per query length using a fixed random seed (seed = 0), ensuring reproducibility and enabling estimation of variability.

## Benchmark procedures

Each method was benchmarked independently. For each replicate, all pairs in that replicate were processed as a batch. Wall-clock time was measured with `time.perf_counter_ns`. The throughput metric recorded is the total number of non-reference allele calls (alt calls) retrieved across all pairs in the batch, divided by the elapsed time in seconds.

For SVAR and PRESUB-BCF, the benchmark separates a **setup phase** (one-time index search or sample subsetting) from a **read phase** (data gather / iteration). The read phase is what is timed in `elapsed_ns` and constitutes the primary throughput metric; the setup phase is timed separately into `setup_ns`. In the intended downstream use case, setup is amortized across many reads and is therefore a secondary metric — informative but not the headline figure.

**SVAR.** Genotypes are stored as a memory-mapped sparse array via `genoray.SparseVar`. For each batch, offset ranges for all (region, sample) pairs are located with a single vectorized index search (`_find_starts_ends`), timed into `setup_ns`. The scattered genotype values are then gathered into a contiguous buffer using a parallelized Numba JIT kernel (`_gather_parallel`, `nopython`, `parallel=True`), timed into `elapsed_ns`.

**BCF.** Genotypes are read sequentially from a CSI-indexed BCF file via `genoray.VCF`. For each (region, sample) pair in the batch, the sample filter is set and the region is queried individually; the full batch loop is timed as a single wall-clock interval (`elapsed_ns`). There is no separate setup phase (`setup_ns` is null).

**PGEN.** Genotypes are read sequentially from a PGEN file via `genoray.PGEN`, using the same per-pair loop structure as BCF. No setup phase.

**PRESUB-BCF.** This condition models a preprocessing-then-read workflow. For each pair, `bcftools view` is invoked as a subprocess to extract the target sample and region into a temporary BCF file (filtering to sites with at least one alt allele, `--min-ac 1`). The total subprocess time across the batch is timed into `setup_ns`. The temporary files are then read with `cyvcf2` and that read time is recorded in `elapsed_ns`. Temporary files are cleaned up after each replicate.

## Memory measurement

Peak resident set size (RSS) is sampled at approximately 20 Hz by a dedicated background thread (`PeakRssSampler`, using recursive `psutil` RSS enumeration over the process tree), wrapping only the **read phase** of each replicate. The setup phase (search, subsetting) is excluded from memory measurement because the downstream use case amortizes it over many reads.

To prevent the RSS sampling thread from perturbing read timing, memory is measured in a **dedicated Nextflow process** that runs in parallel with the throughput process for each (method, query length) combination. The two runs consume identical pair lists but produce separate CSV files (`*_memory.csv` vs. `*_throughput.csv`).

## Hardware and software

All benchmarks ran on a single node (carter-cn-04: AMD EPYC 7543, 2 sockets × 32 cores, 128 hardware threads, AVX2, 32 MiB L3 cache per socket, dual NUMA), with 8 CPUs and 64 GB RAM allocated per benchmark process via Nextflow. The node was pinned with `--nodelist=carter-cn-04` to keep CPU microarchitecture constant across all runs. The Python environment was managed with pixi (Python 3.12). Key dependencies: `genoray`, `numba`, `awkward`, `cyvcf2`, `bcftools`.

## Analysis

Three figures are produced:

1. **Read throughput** (`plot.{png,svg,pdf}`): all four methods, `y = alt calls / sec` for the read phase vs. query length, log–log scale with LOWESS curves.
2. **Setup throughput** (`setup_plot.{png,svg,pdf}`): SVAR and PRESUB-BCF only, `y = alt calls / sec` using `setup_ns` as the time denominator. The ratio of read throughput to setup throughput at a given query length gives the break-even number of reads needed to amortize the setup cost.
3. **Peak memory** (`memory_plot.{png,svg,pdf}`): all four methods, `y = peak RSS (MiB)` for the read phase vs. query length, log–log scale with LOWESS curves.
