nextflow.enable.dsl = 2
nextflow.preview.types = true


params {
    dataset: String
    fasta: Path
    variants: List<Path>
    ds_dir: Path
    bigwig_table: Path?
    min_npb: Integer? = null
    max_npb: Integer = 2 ** 33
    bench_haps: Boolean = true
    bench_tracks: Boolean = false
    bed_dir: Path = projectDir / "beds"
    results_dir: Path = projectDir / "results"
}

workflow {
    if (!params.dataset || !params.fasta || !params.variants) {
        error(
            "Missing required dataset params (dataset, fasta, variants). " + "Provide them via a config file, e.g. 'throughput/tcga-atac.config'."
        )
    }

    if (!params.bigwig_table && params.bench_tracks) {
        log.warn("Ignoring bench_tracks since the dataset '${params.dataset}' does not have bigwig tracks.")
    }

    def lengths: Channel<Integer> = channel.fromList([2048, 16384, 131072, 1048576])

    datasets = WRITE_DATASET(params.dataset, lengths, params.variants, params.bigwig_table).out

    def grid: Channel<Tuple<String, Integer, Path>> = MAKE_GRID(datasets)

    if (params.bench_haps) {
        BENCH_HAPS(grid)
    }
    if (params.bench_tracks) {
        BENCH_TRACKS(grid)
    }
}

process WRITE_DATASET {
    queue 'carter-compute'
    cpus 8
    time 7.d
    memory {
        def special_length: Boolean = (length == 1048576 || length == 2048)
        (dataset == 'UKBB') && special_length ? 256.GB : 32.GB
    }

    input:
    dataset: String
    length: Integer
    variants: List<Path>
    bigwig_table: Path?

    script:
    ds_path = params.ds_dir / "seqlen_${length}.gvl"
    vars = variants.first()

    maybe_just_chr22 = dataset == 'UKBB' ? "--just-chr22" : ""

    """
    mkdir -p ${params.ds_dir}
    
    make_bed.py ${length} ${params.fasta} tile_${length}.bed \\
        --canonical --n-samples 100 \\
        ${maybe_just_chr22}

    genvarloader \\
      ${ds_path} \\
      tile_${length}.bed \\
      --variants=${vars} \\
      --overwrite \\
      ${bigwig_table ? "--bigwig-table=${bigwig_table} \\" : ""}
      ${bigwig_table ? "--track-name read-depth" : ""}
    """

    output:
    out: Tuple<String, Integer> = tuple(ds_path.toString(), length)
}

process MAKE_GRID {
    input:
    (ds_path, length): Tuple<String, Integer>

    script:
    min_npb_arg = params.min_npb != null ? "--min-npb ${params.min_npb}" : ""
    grid_file = "grid_${length}.csv"
    """
    make_launch_grid.py \\
      --length ${length} \\
      --max-npb ${params.max_npb} \\
      ${min_npb_arg} \\
      --output ${grid_file}
    """

    output:
    tuple(ds_path, length, file(grid_file))
}

process BENCH_HAPS {
    clusterOptions '--nodelist=carter-cn-04'
    cpus 64
    memory 32.GB * task.attempt
    maxRetries 3
    errorStrategy task.exitStatus in 137..140 ? 'retry' : 'terminate'

    input:
    (ds_path, length, grid_file): Tuple<String, Integer, Path>

    script:
    ds_name = file(ds_path).parent.baseName
    results = params.results_dir / "haps" / "${ds_name}_${length}.csv"

    """
    mkdir -p ${params.results_dir}/haps
    benchmark_haps.py \\
      ${results} \\
      ${ds_path} \\
      ${length} \\
      ${params.fasta} \\
      ${grid_file}
    """

    output:
    results
}

process BENCH_TRACKS {
    clusterOptions '--nodelist=carter-cn-04'
    cpus 64
    memory 32.GB * task.attempt
    maxRetries 3
    errorStrategy task.exitStatus in 137..140 ? 'retry' : 'terminate'
    maxForks 8

    input:
    (ds_path, length, grid_file): Tuple<String, Integer, Path>

    script:
    ds_name = file(ds_path).parent.baseName
    results = params.results_dir / "tracks" / "${ds_name}_${length}.csv"

    """
    benchmark_tracks.py \\
      ${results} \\
      ${ds_path} \\
      ${length} \\
      ${params.fasta} \\
      ${grid_file}
    """

    output:
    results
}
