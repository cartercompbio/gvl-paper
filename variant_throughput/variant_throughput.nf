#! /usr/bin/env nextflow run

nextflow.enable.types = true

params {
    dataset: String

    svar: Path
    bcf: Path
    pgen: Path
    fai: Path
    seed: Integer = 0
    n_replicates: Integer = 5
    max_pairs: Integer = 100
    max_total_length: Integer = 16777216
    query_lengths: List<Integer> = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216]
    use_custom_pack: Boolean = true
}

workflow {

    main:
    lengths = channel.fromList(params.query_lengths)
    pairs = GENERATE_PAIRS(
        lengths,
        params.n_replicates,
        params.max_pairs,
        params.seed,
        params.max_total_length,
        n_full,
    )

    // Throughput track
    svar_t = BENCH_SVAR_THROUGHPUT(pairs)
    bcf_t = BENCH_BCF_THROUGHPUT(pairs)
    pgen_t = BENCH_PGEN_THROUGHPUT(pairs)
    presub_t = BENCH_PRESUBSET_BCF_THROUGHPUT(pairs)

    // Memory track (runs in parallel with throughput)
    svar_m = BENCH_SVAR_MEMORY(pairs)
    bcf_m = BENCH_BCF_MEMORY(pairs)
    pgen_m = BENCH_PGEN_MEMORY(pairs)
    presub_m = BENCH_PRESUBSET_BCF_MEMORY(pairs)

    throughput_grouped = svar_t
        .mix(bcf_t)
        .mix(pgen_t)
        .mix(presub_t)
        .map { r -> tuple(r.method, r.csv) }
        .groupBy()

    memory_grouped = svar_m
        .mix(bcf_m)
        .mix(pgen_m)
        .mix(presub_m)
        .map { r -> tuple(r.method, r.csv) }
        .groupBy()

    combined_throughput = COMBINE_THROUGHPUT(throughput_grouped)
    combined_memory = COMBINE_MEMORY(memory_grouped)

    throughput_plots = PLOT_THROUGHPUT(combined_throughput.map { r -> r.csv }.collect())
    memory_plots = PLOT_MEMORY(combined_memory.map { r -> r.csv }.collect())

    publish:
    combined_throughput: Channel<MethodResult> = combined_throughput
    combined_memory: Channel<MethodResult> = combined_memory
    throughput_plots: Value<ThroughputPlots> = throughput_plots
    memory_plots: Value<MemoryPlots> = memory_plots
}

output {
    combined_throughput: Channel<MethodResult> {
        path { r -> r.csv >> "${r.method}_throughput.csv" }
    }
    combined_memory: Channel<MethodResult> {
        path { r -> r.csv >> "${r.method}_memory.csv" }
    }
    throughput_plots: Value<ThroughputPlots> {
        path { r ->
            r.plot_png >> "plot.png"
            r.plot_svg >> "plot.svg"
            r.plot_pdf >> "plot.pdf"
            r.setup_png >> "setup_plot.png"
            r.setup_svg >> "setup_plot.svg"
            r.setup_pdf >> "setup_plot.pdf"
        }
    }
    memory_plots: Value<MemoryPlots> {
        path { r ->
            r.png >> "memory_plot.png"
            r.svg >> "memory_plot.svg"
            r.pdf >> "memory_plot.pdf"
        }
    }
}

process GENERATE_PAIRS {
    queue 'carter-compute'
    cpus 2
    time 2.h
    memory 16.GB

    input:
    query_length: Integer
    n_replicates: Integer
    max_pairs: Integer
    seed: Integer
    max_total_length: Integer
    n_samples_full: Integer

    output:
    record(
        query_length: query_length,
        n_samples: n_samples_full,
        pairs: file("pairs_${query_length}.parquet"),
        svar: file("${params.svar}"),
        bcf: file("${params.bcf}"),
        pgen: file("${params.pgen}"),
    )

    script:
    """
    generate_pairs.py \\
      ${params.svar} \\
      ${params.fai} \\
      ${query_length} \\
      pairs_${query_length}.parquet \\
      --seed ${seed} \\
      --n-replicates ${n_replicates} \\
      --max-pairs ${max_pairs} \\
      --max-total-length ${max_total_length}
    """
}

process BENCH_SVAR_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "svar", csv: file("svar_q${p.query_length}_n${p.n_samples}_throughput.csv"))

    script:
    pack_flag = params.use_custom_pack ? "--use-custom-pack" : "--no-use-custom-pack"
    """
    bench_svar.py \\
      ${p.pairs} \\
      ${p.svar} \\
      svar_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      ${pack_flag} \\
      --n-samples ${p.n_samples}
    """
}

process BENCH_SVAR_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "svar", csv: file("svar_q${p.query_length}_n${p.n_samples}_memory.csv"))

    script:
    pack_flag = params.use_custom_pack ? "--use-custom-pack" : "--no-use-custom-pack"
    """
    bench_svar.py \\
      ${p.pairs} \\
      ${p.svar} \\
      svar_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      ${pack_flag} \\
      --n-samples ${p.n_samples}
    """
}

process BENCH_BCF_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "bcf", csv: file("bcf_q${p.query_length}_n${p.n_samples}_throughput.csv"))

    script:
    """
    bench_bcf.py \\
      ${p.pairs} \\
      ${p.bcf} \\
      bcf_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples}
    """
}

process BENCH_BCF_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "bcf", csv: file("bcf_q${p.query_length}_n${p.n_samples}_memory.csv"))

    script:
    """
    bench_bcf.py \\
      ${p.pairs} \\
      ${p.bcf} \\
      bcf_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples}
    """
}

process BENCH_PGEN_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "pgen", csv: file("pgen_q${p.query_length}_n${p.n_samples}_throughput.csv"))

    script:
    """
    bench_pgen.py \\
      ${p.pairs} \\
      ${p.pgen} \\
      pgen_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples}
    """
}

process BENCH_PGEN_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "pgen", csv: file("pgen_q${p.query_length}_n${p.n_samples}_memory.csv"))

    script:
    """
    bench_pgen.py \\
      ${p.pairs} \\
      ${p.pgen} \\
      pgen_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples}
    """
}

process BENCH_PRESUBSET_BCF_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "presubset_bcf", csv: file("presubset_bcf_q${p.query_length}_n${p.n_samples}_throughput.csv"))

    script:
    """
    bench_presubset_bcf.py \\
      ${p.pairs} \\
      ${p.bcf} \\
      presubset_bcf_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples}
    """
}

process BENCH_PRESUBSET_BCF_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

    output:
    record(method: "presubset_bcf", csv: file("presubset_bcf_q${p.query_length}_n${p.n_samples}_memory.csv"))

    script:
    """
    bench_presubset_bcf.py \\
      ${p.pairs} \\
      ${p.bcf} \\
      presubset_bcf_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples}
    """
}

process COMBINE_THROUGHPUT {
    executor 'local'

    input:
    tuple(method: String, csvs: Iterable<Path>)

    script:
    """
    combine_csvs.py ${method}_throughput.csv ${csvs.join(' ')}
    """

    output:
    record(method: method, csv: file("${method}_throughput.csv"))
}

process COMBINE_MEMORY {
    executor 'local'

    input:
    tuple(method: String, csvs: Iterable<Path>)

    script:
    """
    combine_csvs.py ${method}_memory.csv ${csvs.join(' ')}
    """

    output:
    record(method: method, csv: file("${method}_memory.csv"))
}

process PLOT_THROUGHPUT {
    executor 'local'

    input:
    csvs: Iterable<Path>

    script:
    """
    plot_throughput.py ${csvs.join(' ')} --output-dir .
    """

    output:
    record(
        plot_png: file("plot.png"),
        plot_svg: file("plot.svg"),
        plot_pdf: file("plot.pdf"),
        setup_png: file("setup_plot.png"),
        setup_svg: file("setup_plot.svg"),
        setup_pdf: file("setup_plot.pdf"),
    )
}

process PLOT_MEMORY {
    executor 'local'

    input:
    csvs: Iterable<Path>

    script:
    """
    plot_memory.py ${csvs.join(' ')} --output-dir .
    """

    output:
    record(png: file("memory_plot.png"), svg: file("memory_plot.svg"), pdf: file("memory_plot.pdf"))
}

record ThroughputPlots {
    plot_png: Path
    plot_svg: Path
    plot_pdf: Path
    setup_png: Path
    setup_svg: Path
    setup_pdf: Path
}

record MemoryPlots {
    png: Path
    svg: Path
    pdf: Path
}

record SweepInput {
    query_length: Integer
    n_samples: Integer
    pairs: Path
    svar: Path
    bcf: Path
    pgen: Path
}

record MethodResult {
    method: String
    csv: Path
}
