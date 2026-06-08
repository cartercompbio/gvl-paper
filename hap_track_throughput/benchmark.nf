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
    bench_native: Boolean = false
    no_symbolic: Boolean = true
    no_breakend: Boolean = true
    measure_memory: Boolean = false
    // Per-cell buffered-loader buffer cap (bytes). The loader is double-buffered,
    // so a batch needs buffer >= 2*batch_bytes; benchmark_{haps,tracks}.py size the
    // buffer per cell up to this cap (cells needing more are recorded NaN). 64 GiB
    // covers the full haps range and tracks through saturation. See compare doc.
    max_buffer_bytes: Integer = 68719476736
    // Memory pass emits an RSS-vs-time growth curve (largest-batch operating
    // point) over this many seconds, instead of a single peak/avg row.
    growth_time_s: Integer = 180
    results_dir: String = "${projectDir}/../results_gvl027"
    // When set (memory pass), BENCH_HAPS/BENCH_TRACKS read a derived
    // best-throughput one-row grid from here instead of make_launch_grid.py.
    best_grid_dir: String? = null
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
    // Buffered-only: the default `to_dataloader` (mode=none) is not benchmarked.
    // Buffered is what the manuscript reports; its throughput is amortized over
    // the buffer (per-minibatch torch-collate overhead dominates at tiny batch
    // sizes, so the grid sweeps batch_size). See compare_to_baseline.py.
    dl_modes = channel.fromList(["buffered"])

    // SVAR conversion bench (runs once, not per seqlen). Hap-safe filtering
    // (drop symbolic + breakend ALTs) is applied here so every SVAR is
    // GVL-compatible.
    svar_result = BENCH_SVAR_CONVERT(params.dataset, params.variants)

    // SVAR write inputs: cartesian product of lengths x the single converted svar.
    svar_write_inputs = lengths
        .combine(svar_result.map { r -> r.svar })
        .map { len, svar -> record(length: len, variants: [svar] as List<Path>, backend: "svar") }

    // Native write inputs (raw variants, no SVAR) — gated off by default.
    native_write_inputs = params.bench_native
        ? lengths.map { len -> record(length: len, variants: params.variants, backend: "native") }
        : channel.empty()

    all_write_inputs = svar_write_inputs.mix(native_write_inputs)

    // Dataset write bench (per length x backend)
    write_results = BENCH_WRITE_DATASET(all_write_inputs, params.bigwig_table, params.region)

    // Cross each written dataset with both dataloader modes. The same .gvl is
    // reused across dl_modes (write does not depend on dl_mode).
    dl_inputs = write_results
        .combine(dl_modes)
        .map { wr, dl ->
            record(length: wr.length, backend: wr.backend, gvl: wr.gvl, dl_mode: dl)
        }

    haps_results = params.bench_haps ? BENCH_HAPS(dl_inputs) : channel.empty()
    do_tracks = params.bench_tracks && params.bigwig_table != null
    tracks_results = do_tracks ? BENCH_TRACKS(dl_inputs) : channel.empty()

    publish:
    svar_result: Channel<SvarConvertResult> = svar_result
    write_results: Channel<BenchWriteResult> = write_results
    haps_results: Channel<BenchResult> = haps_results
    tracks_results: Channel<BenchResult> = tracks_results
}

output {
    svar_result: Channel<SvarConvertResult> {
        path { r ->
            r.convert_csv >> "${params.results_dir}/${params.measure_memory ? 'svar_convert_memory' : 'svar_convert'}/${params.dataset}.csv"
        }
    }
    write_results: Channel<BenchWriteResult> {
        path { r ->
            r.gvl >> "${params.ds_dir}/${r.backend}/seqlen_${r.length}.gvl"
            r.write_csv >> "${params.results_dir}/${params.measure_memory ? 'write_memory' : 'write'}/${params.dataset}_${r.length}_${r.backend}.csv"
        }
    }
    haps_results: Channel<BenchResult> {
        path { r ->
            r.csv >> "${params.results_dir}/${params.measure_memory ? 'haps_memory' : 'haps'}/${params.dataset}_${r.length}_${r.backend}_${r.dl_mode}.csv"
        }
    }
    tracks_results: Channel<BenchResult> {
        path { r ->
            r.csv >> "${params.results_dir}/${params.measure_memory ? 'tracks_memory' : 'tracks'}/${params.dataset}_${r.length}_${r.backend}_${r.dl_mode}.csv"
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

    script:
    vars = variants.first()
    mem_flag = params.measure_memory ? "--measure-memory" : ""
    sym_flag = params.no_symbolic ? "--no-symbolic" : "--no-no-symbolic"
    bnd_flag = params.no_breakend ? "--no-breakend" : "--no-no-breakend"
    """
    benchmark_svar_convert.py \\
      svar_convert.csv \\
      ${vars} \\
      output.svar \\
      --dataset ${dataset} \\
      --max-mem 64g \\
      --n-jobs ${task.cpus} \\
      ${sym_flag} \\
      ${bnd_flag} \\
      ${mem_flag}
    """

    output:
    record(convert_csv: file("svar_convert.csv"), svar: file("output.svar"))
}

process BENCH_WRITE_DATASET {
    queue 'carter-compute'
    cpus 8
    time 7.d
    // UKBB has ~487k samples, so writing genotypes is memory-heavy at every
    // seqlen (32G OOM'd seqlen 131072). Give UKBB a high floor and scale on OOM
    // retry; writes are unpinned so big-mem jobs land on cn-02/cn-04 (~950 GB).
    memory { (params.dataset == 'UKBB' ? 256.GB : 32.GB) * task.attempt }
    maxRetries 2
    errorStrategy { task.exitStatus in 137..140 ? 'retry' : 'terminate' }

    input:
    wi: WriteInput
    bigwig_table: Path?
    region: String?

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

    output:
    record(
        length: wi.length,
        backend: wi.backend,
        gvl: file("seqlen_${wi.length}_${wi.backend}.gvl"),
        write_csv: file("write_${wi.length}_${wi.backend}.csv"),
    )
}

process BENCH_HAPS {
    clusterOptions '--nodelist=carter-cn-03'
    cpus params.test_grid ? 8 : 32
    memory 96.GB * task.attempt
    maxRetries 3
    errorStrategy task.exitStatus in 137..140 ? 'retry' : 'terminate'

    input:
    ds: Dataset

    script:
    min_npb_arg = params.min_npb != null ? "--min-npb ${params.min_npb}" : ""
    test_arg = params.test_grid ? "--test" : ""
    mem_flag = params.measure_memory ? "--measure-memory --memory-timeseries --growth-time-s ${params.growth_time_s}" : ""
    use_best = params.measure_memory && params.best_grid_dir != null
    make_grid = use_best ?
        "cp ${params.best_grid_dir}/${params.dataset}_${ds.length}_haps.csv grid_${ds.length}.csv" :
        "make_launch_grid.py ${ds.length} --max-npb ${params.max_npb} ${min_npb_arg} ${test_arg} --output grid_${ds.length}.csv"
    """
    ${make_grid}

    benchmark_haps.py \\
      results_${ds.length}_${ds.backend}_${ds.dl_mode}.csv \\
      ${ds.gvl} \\
      ${ds.length} \\
      ${params.fasta} \\
      grid_${ds.length}.csv \\
      --dataset ${params.dataset} \\
      --backend ${ds.backend} \\
      --dl-mode ${ds.dl_mode} \\
      --max-buffer-bytes ${params.max_buffer_bytes} \\
      ${mem_flag}
    """

    output:
    record(length: ds.length, backend: ds.backend, dl_mode: ds.dl_mode, csv: file("results_${ds.length}_${ds.backend}_${ds.dl_mode}.csv"))
}

process BENCH_TRACKS {
    clusterOptions '--nodelist=carter-cn-03'
    cpus params.test_grid ? 8 : 32
    memory 96.GB * task.attempt
    maxRetries 3
    errorStrategy task.exitStatus in 137..140 ? 'retry' : 'terminate'
    maxForks 8

    input:
    ds: Dataset

    script:
    min_npb_arg = params.min_npb != null ? "--min-npb ${params.min_npb}" : ""
    test_arg = params.test_grid ? "--test" : ""
    mem_flag = params.measure_memory ? "--measure-memory --memory-timeseries --growth-time-s ${params.growth_time_s}" : ""
    use_best = params.measure_memory && params.best_grid_dir != null
    make_grid = use_best ?
        "cp ${params.best_grid_dir}/${params.dataset}_${ds.length}_tracks.csv grid_${ds.length}.csv" :
        "make_launch_grid.py ${ds.length} --max-npb ${params.max_npb} ${min_npb_arg} ${test_arg} --output grid_${ds.length}.csv"
    """
    ${make_grid}

    benchmark_tracks.py \\
      results_${ds.length}_${ds.backend}_${ds.dl_mode}.csv \\
      ${ds.gvl} \\
      ${ds.length} \\
      ${params.fasta} \\
      grid_${ds.length}.csv \\
      --dataset ${params.dataset} \\
      --backend ${ds.backend} \\
      --dl-mode ${ds.dl_mode} \\
      --max-buffer-bytes ${params.max_buffer_bytes} \\
      ${mem_flag}
    """

    output:
    record(length: ds.length, backend: ds.backend, dl_mode: ds.dl_mode, csv: file("results_${ds.length}_${ds.backend}_${ds.dl_mode}.csv"))
}

record Dataset {
    length: Integer
    backend: String
    dl_mode: String
    gvl: Path
}

record WriteInput {
    length: Integer
    variants: List<Path>
    backend: String
}

record BenchResult {
    length: Integer
    backend: String
    dl_mode: String
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
