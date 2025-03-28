# GenVarLoader Paper
This repository contains results and code pertaining to the GenVarLoader manuscript. The [GenVarLoader](https://github.com/mcvickerlab/GenVarLoader) package itself is available from PyPI.

# Data Availability
Publically available GVL datasets from the 1000 Genomes Project are available from [Zenodo](https://doi.org/10.5281/zenodo.14367502). All other data from the paper are controlled access.

# Running the 1kGP benchmark
To run benchmarks for the 1000 Genomes Project, download the tar archvies from [Zenodo](https://doi.org/10.5281/zenodo.14367502) and extract them into `./throughput/datasets/1kgp`. Then, some manual editing of the SLURM commands in `./throughput/launch_benchmarks.py` may be required depending on whether SLURM is available and/or node names. In addition, the reference genome used by the 1000 Genomes Project should be downloaded to `./throughput/` from their [FTP site](ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa).
