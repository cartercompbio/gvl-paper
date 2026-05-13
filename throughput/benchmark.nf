#! /usr/bin/env nextflow run

nextflow.enable.dsl = 2
nextflow.enable.types = true


params {
    dataset: String
    fasta: Path
    variants: List<Path>
    ds_dir: String
    bigwig_table: Path?
    region: String?
    min_npb: Integer? = null
    max_npb: Integer = 2 ** 33
    test_grid: Boolean = false
    bench_haps: Boolean = true
    bench_tracks: Boolean = false
    measure_memory: Boolean = false
    results_dir: String = "${projectDir}/results"
}

record Dataset {
    length: Integer
    gvl: Path
}

record Grid {
    length: Integer
    grid: Path
}

record GridSpec {
    length: Integer
    gvl: Path
    grid: Path
}

record BenchResult {
    length: Integer
    csv: Path
}

workflow {
    main:
    if (!params.dataset || !params.fasta || !params.variants) {
        error(
            "Missing required dataset params (dataset, fasta, variants). " + "Provide them via a config file, e.g. 'throughput/tcga-atac.config'."
        )
    }

    if (!params.bigwig_table && params.bench_tracks) {
        log.warn("Ignoring bench_tracks since the dataset '${params.dataset}' does not have bigwig tracks.")
    }

    lengths = channel.fromList([2048, 16384, 131072, 1048576])

    datasets = WRITE_DATASET(params.dataset, lengths, params.variants, params.bigwig_table, params.region)

    grids = MAKE_GRID(datasets)
    specs = datasets.join(grids, by: 'length')

    haps_results = params.bench_haps ? BENCH_HAPS(specs) : channel.empty()
    do_tracks = params.bench_tracks && params.bigwig_table != null
    tracks_results = do_tracks ? BENCH_TRACKS(specs) : channel.empty()

    publish:
    datasets = datasets
    haps_results = haps_results
    tracks_results = tracks_results
}

output {
    datasets: Channel<Dataset> {
        path { d ->
            d.gvl >> "${params.ds_dir}/seqlen_${d.length}.gvl"
        }
    }
    haps_results: Channel<BenchResult> {
        path { r ->
            def sub = params.measure_memory ? "haps_memory" : "haps"
            r.csv >> "${params.results_dir}/${sub}/${params.dataset}_${r.length}.csv"
        }
    }
    tracks_results: Channel<BenchResult> {
        path { r ->
            def sub = params.measure_memory ? "tracks_memory" : "tracks"
            r.csv >> "${params.results_dir}/${sub}/${params.dataset}_${r.length}.csv"
        }
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
    region: String?

    output:
    record(length: length, gvl: file("seqlen_${length}.gvl"))

    script:
    vars = variants.first()
    maybe_region = region ? "--region ${region}" : ""

    """
    make_bed.py ${length} ${params.fasta} tile_${length}.bed \\
        --canonical --n-samples 100 \\
        ${maybe_region}

    genvarloader \\
      seqlen_${length}.gvl \\
      tile_${length}.bed \\
      --variants=${vars} \\
      --overwrite \\
      ${bigwig_table ? "--bigwig-table=${bigwig_table} \\" : ""}
      ${bigwig_table ? "--track-name read-depth" : ""}
    """
}

process MAKE_GRID {
    input:
    ds: Dataset

    output:
    record(length: ds.length, grid: file("grid_${ds.length}.csv"))

    script:
    min_npb_arg = params.min_npb != null ? "--min-npb ${params.min_npb}" : ""
    test_arg = params.test_grid ? "--test" : ""
    """
    make_launch_grid.py \\
      ${ds.length} \\
      --max-npb ${params.max_npb} \\
      ${min_npb_arg} \\
      ${test_arg} \\
      --output grid_${ds.length}.csv
    """
}

process BENCH_HAPS {
    clusterOptions '--nodelist=carter-cn-04'
    cpus 64
    memory 32.GB * task.attempt
    maxRetries 3
    errorStrategy task.exitStatus in 137..140 ? 'retry' : 'terminate'

    input:
    spec: GridSpec

    output:
    record(length: spec.length, csv: file("results_${spec.length}.csv"))

    script:
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    """
    benchmark_haps.py \\
      results_${spec.length}.csv \\
      ${spec.gvl} \\
      ${spec.length} \\
      ${params.fasta} \\
      ${spec.grid} \\
      --dataset ${params.dataset} ${mem_flag}
    """
}

process BENCH_TRACKS {
    clusterOptions '--nodelist=carter-cn-04'
    cpus 64
    memory 32.GB * task.attempt
    maxRetries 3
    errorStrategy task.exitStatus in 137..140 ? 'retry' : 'terminate'
    maxForks 8

    input:
    spec: GridSpec

    output:
    record(length: spec.length, csv: file("results_${spec.length}.csv"))

    script:
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    """
    benchmark_tracks.py \\
      results_${spec.length}.csv \\
      ${spec.gvl} \\
      ${spec.length} \\
      ${params.fasta} \\
      ${spec.grid} \\
      --dataset ${params.dataset} ${mem_flag}
    """
}
