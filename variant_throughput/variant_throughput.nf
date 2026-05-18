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
    sample_sizes: List<Integer> = [10, 32, 100, 316, 1000, 3202]
    n_sweep_query_length: Integer = 131072
    sample_seed: Integer = 0
}

workflow {

    main:
    n_full = COUNT_FULL_SAMPLES(params.svar)
    lengths = channel.fromList(params.query_lengths)
    pairs_raw = GENERATE_PAIRS(
        lengths,
        params.n_replicates,
        params.max_pairs,
        params.seed,
        params.max_total_length,
        n_full,
    )
    pairs = pairs_raw.map { r ->
        record(
            query_length: r.query_length,
            n_samples: r.n_samples,
            pairs: r.pairs,
            svar: params.svar,
            bcf: params.bcf,
            pgen: params.pgen,
        ) as SweepInput
    }

    n_channel = channel.fromList(params.sample_sizes)

    sample_lists = MAKE_SAMPLE_LIST(n_channel, params.sample_seed, params.svar)

    pgen_stem = params.pgen.toString().replaceAll(/\.pgen$/, '')
    pvar_path = file("${pgen_stem}.pvar")
    psam_path = file("${pgen_stem}.psam")

    subset_bcf_out  = SUBSET_BCF (sample_lists.map { r -> r.n }, sample_lists.map { r -> r.samples }, params.bcf)
    subset_pgen_out = SUBSET_PGEN(
        sample_lists.map { r -> r.n },
        sample_lists.map { r -> r.samples },
        params.pgen,
        pvar_path,
        psam_path,
    )
    svar_out        = BUILD_SVAR_FROM_PGEN(
        subset_pgen_out.map { r -> r.n },
        subset_pgen_out.map { r -> r.pgen },
        subset_pgen_out.map { r -> r.pvar },
        subset_pgen_out.map { r -> r.psam },
    )

    triples = svar_out
        .map { r -> tuple(r.n, r.svar) }
        .join(subset_bcf_out.map { r -> tuple(r.n, r.bcf) },  by: 0)
        .join(subset_pgen_out.map { r -> tuple(r.n, r.pgen) }, by: 0)
        .map { n, svar, bcf, pgen ->
            record(n: n, svar: svar, bcf: bcf, pgen: pgen) as SubsetTriple
        }

    n_pairs = GENERATE_PAIRS_N(
        triples,
        params.n_sweep_query_length,
        params.n_replicates,
        params.max_pairs,
        params.seed,
        params.max_total_length,
    )

    all_inputs = pairs.mix(n_pairs)

    // Throughput track
    svar_t = BENCH_SVAR_THROUGHPUT(all_inputs)
    bcf_t = BENCH_BCF_THROUGHPUT(all_inputs)
    pgen_t = BENCH_PGEN_THROUGHPUT(all_inputs)
    presub_t = BENCH_PRESUBSET_BCF_THROUGHPUT(all_inputs)

    // Memory track (runs in parallel with throughput)
    svar_m = BENCH_SVAR_MEMORY(all_inputs)
    bcf_m = BENCH_BCF_MEMORY(all_inputs)
    pgen_m = BENCH_PGEN_MEMORY(all_inputs)
    presub_m = BENCH_PRESUBSET_BCF_MEMORY(all_inputs)

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
            r.n_plot_png >> "n_plot.png"
            r.n_plot_svg >> "n_plot.svg"
            r.n_plot_pdf >> "n_plot.pdf"
        }
    }
    memory_plots: Value<MemoryPlots> {
        path { r ->
            r.png >> "memory_plot.png"
            r.svg >> "memory_plot.svg"
            r.pdf >> "memory_plot.pdf"
            r.n_png >> "n_memory_plot.png"
            r.n_svg >> "n_memory_plot.svg"
            r.n_pdf >> "n_memory_plot.pdf"
        }
    }
}

process COUNT_FULL_SAMPLES {
    queue 'carter-compute'
    cpus 1
    time 30.min
    memory 32.GB

    input:
    svar: Path

    output:
    n: Integer = stdout().trim().toInteger()

    script:
    """
    make_sample_list.py ${svar} /dev/null --print-total
    """
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

process MAKE_SAMPLE_LIST {
    queue 'carter-compute'
    cpus 1
    time 30.min
    memory 32.GB

    input:
    n: Integer
    seed: Integer
    svar: Path

    output:
    record(n: n, samples: file("samples_N${n}.txt"))

    script:
    """
    make_sample_list.py ${svar} samples_N${n}.txt --n ${n} --seed ${seed}
    """
}

process SUBSET_BCF {
    queue 'carter-compute'
    cpus 4
    time 4.h
    memory 16.GB

    input:
    n: Integer
    samples: Path
    bcf: Path

    output:
    record(n: n, bcf: file("N${n}.bcf"), csi: file("N${n}.bcf.csi"))

    script:
    """
    bcftools view -S ${samples} --force-samples --no-update --threads ${task.cpus} -Ob -o N${n}.bcf ${bcf}
    bcftools index --threads ${task.cpus} N${n}.bcf
    """
}

process SUBSET_PGEN {
    queue 'carter-compute'
    cpus 4
    time 4.h
    memory 16.GB

    input:
    n: Integer
    samples: Path
    pgen: Path
    pvar: Path
    psam: Path

    stage:
    stageAs pgen, 'in.pgen'
    stageAs pvar, 'in.pvar'
    stageAs psam, 'in.psam'

    output:
    record(
        n: n,
        pgen: file("N${n}.pgen"),
        pvar: file("N${n}.pvar"),
        psam: file("N${n}.psam"),
    )

    script:
    """
    awk 'BEGIN{OFS="\\t"} {print "0", \$1}' ${samples} > keep.tsv
    plink2 --pfile in --keep keep.tsv --make-pgen --threads ${task.cpus} --out N${n}
    """
}

process BUILD_SVAR_FROM_PGEN {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 8.h
    memory 64.GB

    input:
    n: Integer
    pgen: Path
    pvar: Path
    psam: Path

    output:
    record(n: n, svar: file("N${n}.svar"))

    script:
    """
    python - <<'PY'
from pathlib import Path
from genoray import SparseVar
SparseVar.from_pgen(Path("${pgen}"), Path("N${n}.svar"))
PY
    """
}

process GENERATE_PAIRS_N {
    queue 'carter-compute'
    cpus 2
    time 2.h
    memory 16.GB

    input:
    t: SubsetTriple
    query_length: Integer
    n_replicates: Integer
    max_pairs: Integer
    seed: Integer
    max_total_length: Integer

    output:
    record(
        query_length: query_length,
        n_samples: t.n,
        pairs: file("pairs_N${t.n}.parquet"),
        svar: t.svar,
        bcf: t.bcf,
        pgen: t.pgen,
    )

    script:
    """
    generate_pairs.py \\
      ${t.svar} \\
      ${params.fai} \\
      ${query_length} \\
      pairs_N${t.n}.parquet \\
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
        n_plot_png: file("n_plot.png"),
        n_plot_svg: file("n_plot.svg"),
        n_plot_pdf: file("n_plot.pdf"),
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
    record(
        png: file("memory_plot.png"),
        svg: file("memory_plot.svg"),
        pdf: file("memory_plot.pdf"),
        n_png: file("n_memory_plot.png"),
        n_svg: file("n_memory_plot.svg"),
        n_pdf: file("n_memory_plot.pdf"),
    )
}

record ThroughputPlots {
    plot_png: Path
    plot_svg: Path
    plot_pdf: Path
    setup_png: Path
    setup_svg: Path
    setup_pdf: Path
    n_plot_png: Path
    n_plot_svg: Path
    n_plot_pdf: Path
}

record MemoryPlots {
    png: Path
    svg: Path
    pdf: Path
    n_png: Path
    n_svg: Path
    n_pdf: Path
}

record SubsetTriple {
    n: Integer
    svar: Path
    bcf: Path
    pgen: Path
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
