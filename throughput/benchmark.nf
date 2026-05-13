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
    backend: String
    gvl: Path
}

record WriteInput {
    length: Integer
    variants: List<Path>
    backend: String
}

record GridSpec {
    length: Integer
    backend: String
    gvl: Path
    grid: Path
}

record BenchResult {
    length: Integer
    backend: String
    csv: Path
}

record SvarConvertResult {
    convert_csv: Path
    svar: Path
}

record BenchWriteResult {
    length: Integer
    backend: String
    gvl: Path
    write_csv: Path
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

    // SVAR conversion bench (runs once, not per seqlen)
    svar_result = BENCH_SVAR_CONVERT(params.dataset, params.variants)

    // Native write inputs: one per length, using raw variants directly
    native_write_inputs = lengths.map { len -> record(length: len, variants: params.variants, backend: "native") }

    // SVAR write inputs: cartesian product of lengths x the single converted svar
    svar_write_inputs = lengths
        .combine(svar_result.map { r -> r.svar })
        .map { len, svar -> record(length: len, variants: [svar] as List<Path>, backend: "svar") }

    all_write_inputs = native_write_inputs.mix(svar_write_inputs)

    // Dataset write bench (per length x backend)
    write_results = BENCH_WRITE_DATASET(all_write_inputs, params.bigwig_table, params.region)

    // MAKE_GRID takes Dataset and outputs GridSpec (passes gvl through)
    datasets = write_results.map { r -> record(length: r.length, backend: r.backend, gvl: r.gvl) }
    specs = MAKE_GRID(datasets)

    haps_results = params.bench_haps ? BENCH_HAPS(specs) : channel.empty()
    do_tracks = params.bench_tracks && params.bigwig_table != null
    tracks_results = do_tracks ? BENCH_TRACKS(specs) : channel.empty()

    publish:
    svar_result = svar_result
    write_results = write_results
    haps_results = haps_results
    tracks_results = tracks_results
}

output {
    svar_result: Channel<SvarConvertResult> {
        path { r ->
            def sub = params.measure_memory ? "svar_convert_memory" : "svar_convert"
            r.convert_csv >> "${params.results_dir}/${sub}/${params.dataset}.csv"
        }
    }
    write_results: Channel<BenchWriteResult> {
        path { r ->
            def sub = params.measure_memory ? "write_memory" : "write"
            r.gvl >> "${params.ds_dir}/${r.backend}/seqlen_${r.length}.gvl"
            r.write_csv >> "${params.results_dir}/${sub}/${params.dataset}_${r.length}_${r.backend}.csv"
        }
    }
    haps_results: Channel<BenchResult> {
        path { r ->
            def sub = params.measure_memory ? "haps_memory" : "haps"
            r.csv >> "${params.results_dir}/${sub}/${params.dataset}_${r.length}_${r.backend}.csv"
        }
    }
    tracks_results: Channel<BenchResult> {
        path { r ->
            def sub = params.measure_memory ? "tracks_memory" : "tracks"
            r.csv >> "${params.results_dir}/${sub}/${params.dataset}_${r.length}_${r.backend}.csv"
        }
    }
}

process BENCH_SVAR_CONVERT {
    queue 'carter-compute'
    cpus 8
    time 7.d
    memory 128.GB

    input:
    dataset: String
    variants: List<Path>

    output:
    record(convert_csv: file("svar_convert.csv"), svar: file("output.svar"))

    script:
    vars = variants.first()
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    """
    benchmark_svar_convert.py \\
      svar_convert.csv \\
      ${vars} \\
      output.svar \\
      --dataset ${dataset} \\
      --max-mem 64g \\
      --n-jobs ${task.cpus} \\
      ${mem_flag}
    """
}

process BENCH_WRITE_DATASET {
    queue 'carter-compute'
    cpus 8
    time 7.d
    memory {
        def special_length: Boolean = (wi.length == 1048576 || wi.length == 2048)
        (params.dataset == 'UKBB') && special_length ? 256.GB : 32.GB
    }

    input:
    wi: WriteInput
    bigwig_table: Path?
    region: String?

    output:
    record(
        length: wi.length,
        backend: wi.backend,
        gvl: file("seqlen_${wi.length}_${wi.backend}.gvl"),
        write_csv: file("write_${wi.length}_${wi.backend}.csv")
    )

    script:
    vars = wi.variants.first()
    maybe_region = region ? "--region ${region}" : ""
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    maybe_bigwig = bigwig_table ? "--bigwig-table=${bigwig_table}" : ""
    """
    make_bed.py ${wi.length} ${params.fasta} tile_${wi.length}.bed \\
        --canonical --n-samples 100 \\
        ${maybe_region}

    benchmark_write.py \\
      write_${wi.length}_${wi.backend}.csv \\
      ${vars} \\
      tile_${wi.length}.bed \\
      ${params.fasta} \\
      seqlen_${wi.length}_${wi.backend}.gvl \\
      ${wi.length} \\
      ${wi.backend} \\
      --dataset ${params.dataset} \\
      --max-mem 16g \\
      ${maybe_bigwig} \\
      ${mem_flag}
    """
}

process MAKE_GRID {
    input:
    ds: Dataset

    output:
    record(length: ds.length, backend: ds.backend, gvl: ds.gvl, grid: file("grid_${ds.length}.csv"))

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
    cpus params.test_grid ? 8 : 64
    memory 32.GB * task.attempt
    maxRetries 3
    errorStrategy task.exitStatus in 137..140 ? 'retry' : 'terminate'

    input:
    spec: GridSpec

    output:
    record(length: spec.length, backend: spec.backend, csv: file("results_${spec.length}_${spec.backend}.csv"))

    script:
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    """
    benchmark_haps.py \\
      results_${spec.length}_${spec.backend}.csv \\
      ${spec.gvl} \\
      ${spec.length} \\
      ${params.fasta} \\
      ${spec.grid} \\
      --dataset ${params.dataset} \\
      --backend ${spec.backend} \\
      ${mem_flag}
    """
}

process BENCH_TRACKS {
    clusterOptions '--nodelist=carter-cn-04'
    cpus params.test_grid ? 8 : 64
    memory 32.GB * task.attempt
    maxRetries 3
    errorStrategy task.exitStatus in 137..140 ? 'retry' : 'terminate'
    maxForks 8

    input:
    spec: GridSpec

    output:
    record(length: spec.length, backend: spec.backend, csv: file("results_${spec.length}_${spec.backend}.csv"))

    script:
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    """
    benchmark_tracks.py \\
      results_${spec.length}_${spec.backend}.csv \\
      ${spec.gvl} \\
      ${spec.length} \\
      ${params.fasta} \\
      ${spec.grid} \\
      --dataset ${params.dataset} \\
      --backend ${spec.backend} \\
      ${mem_flag}
    """
}
