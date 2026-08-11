#! /usr/bin/env nextflow run

nextflow.enable.types = true

params {
    dataset: String

    svar: Path
    svar2: Path
    bcf: Path
    pgen: Path
    fai: Path
    seed: Integer = 0
    n_replicates: Integer = 5
    stream_batches: Integer = 8
    bp_budget: Integer = 16777216
    min_seconds: Double = 5.0d
    min_batches: Integer = 10
    query_lengths: List<Integer> = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216]
    use_custom_pack: Boolean = true
    sample_sizes: List<Integer> = [10, 32, 100, 316, 1000, 3202]
    n_sweep_query_length: Integer = 131072
    sample_seed: Integer = 0
}

workflow {

    main:
    n_full = COUNT_FULL_SAMPLES(params.svar)

    bcf_csi_path = file("${params.bcf}.csi")
    pgen_stem = params.pgen.toString().replaceAll(/\.pgen$/, '')
    pvar_path = file("${pgen_stem}.pvar")
    psam_path = file("${pgen_stem}.psam")
    // Pre-built genoray PGEN random-access index for the full cohort, already
    // on disk (6.54GB, 2025-04-21) -- carried on every SweepInput's `gvi`
    // field (see `pairs` below) so genoray's _valid_index() finds it and
    // loads it (~23s, verified directly against this exact file with
    // genoray 2.9.0 -- the version bench_pgen.py actually imports) instead
    // of rebuilding one from scratch (~28-29min measured in the smoke run,
    // since Nextflow stages .pgen/.pvar/.psam as fresh per-task symlinks and
    // this .gvi sidecar was never among them).
    //
    // The cohort-sweep's per-n subset pgens (below) get their OWN `gvi`,
    // built once by BUILD_SVAR_FROM_PGEN (which already constructs a PGEN
    // reader as a side effect of `genoray write`) and threaded through
    // SubsetStores/GENERATE_PAIRS_N -- NOT this full-cohort file, which
    // would silently mismatch a subset's different variant set (genoray's
    // _valid_index only checks existence + mtime ordering, not content).
    // Before that dedup, BUILD_SVAR_FROM_PGEN, BENCH_PGEN_THROUGHPUT, and
    // BENCH_PGEN_MEMORY each independently rebuilt their own copy of a
    // subset's index (up to 20GB+ each even for the smallest n=32 subset,
    // confirmed from the failed campaign's disk footprint) -- a second,
    // independent driver of the ENOSPC failure alongside the uncompressed
    // .pvar text SUBSET_PGEN used to emit (see SUBSET_PGEN below).
    pvar_gvi_path = file("${pvar_path}.gvi")

    lengths = channel.fromList(params.query_lengths)
    pairs_raw = GENERATE_PAIRS(
        lengths,
        params.n_replicates,
        params.stream_batches,
        params.seed,
        params.bp_budget,
        n_full,
    )
    pairs = pairs_raw.map { r ->
        record(
            query_length: r.query_length,
            n_samples: r.n_samples,
            pairs: r.pairs,
            svar: params.svar,
            svar2: params.svar2,
            bcf: params.bcf,
            bcf_csi: bcf_csi_path,
            pgen: params.pgen,
            pvar: pvar_path,
            psam: psam_path,
            gvi: pvar_gvi_path,
        ) as SweepInput
    }

    n_channel = channel.fromList(params.sample_sizes)

    sample_lists = MAKE_SAMPLE_LIST(n_channel, params.sample_seed, params.svar)

    subset_bcf_out = SUBSET_BCF(sample_lists.map { r -> r.n }, sample_lists.map { r -> r.samples }, params.bcf)
    subset_pgen_out = SUBSET_PGEN(
        sample_lists.map { r -> r.n },
        sample_lists.map { r -> r.samples },
        params.pgen,
        pvar_path,
        psam_path,
    )
    svar_out = BUILD_SVAR_FROM_PGEN(
        subset_pgen_out.map { r -> r.n },
        subset_pgen_out.map { r -> r.pgen },
        subset_pgen_out.map { r -> r.pvar },
        subset_pgen_out.map { r -> r.psam },
    )
    svar2_out = BUILD_SVAR2_FROM_PGEN(
        subset_pgen_out.map { r -> r.n },
        subset_pgen_out.map { r -> r.pgen },
        subset_pgen_out.map { r -> r.pvar },
        subset_pgen_out.map { r -> r.psam },
    )

    subset_stores = svar_out
        // svar_out.gvi: BUILD_SVAR_FROM_PGEN's own genoray PGEN index for
        // this n's subset pgen, built once and reused (see pvar_gvi_path
        // above and BUILD_SVAR_FROM_PGEN's output below) instead of being
        // independently rebuilt by every downstream consumer.
        .map { r -> tuple(r.n, r.svar, r.gvi) }
        .join(svar2_out.map { r -> tuple(r.n, r.svar2) }, by: 0)
        .join(subset_bcf_out.map { r -> tuple(r.n, r.bcf, r.csi) }, by: 0)
        .join(subset_pgen_out.map { r -> tuple(r.n, r.pgen, r.pvar, r.psam) }, by: 0)
        .map { n, svar, gvi, svar2, bcf, csi, pgen, pvar, psam ->
            record(
                n: n,
                svar: svar,
                svar2: svar2,
                bcf: bcf,
                bcf_csi: csi,
                pgen: pgen,
                pvar: pvar,
                psam: psam,
                pvar_gvi: gvi,
            ) as SubsetStores
        }

    n_pairs = GENERATE_PAIRS_N(
        subset_stores,
        params.n_sweep_query_length,
        params.n_replicates,
        params.stream_batches,
        params.seed,
        params.bp_budget,
    )

    all_inputs = pairs.mix(n_pairs)

    // Throughput track
    svar_t = BENCH_SVAR_THROUGHPUT(all_inputs)
    svar2_t = BENCH_SVAR2_THROUGHPUT(all_inputs)
    bcf_t = BENCH_BCF_THROUGHPUT(all_inputs)
    pgen_t = BENCH_PGEN_THROUGHPUT(all_inputs)
    presub_t = BENCH_PRESUBSET_BCF_THROUGHPUT(all_inputs)

    // Memory track (runs in parallel with throughput)
    svar_m = BENCH_SVAR_MEMORY(all_inputs)
    svar2_m = BENCH_SVAR2_MEMORY(all_inputs)
    bcf_m = BENCH_BCF_MEMORY(all_inputs)
    pgen_m = BENCH_PGEN_MEMORY(all_inputs)
    presub_m = BENCH_PRESUBSET_BCF_MEMORY(all_inputs)

    throughput_grouped = svar_t
        .mix(svar2_t)
        .mix(bcf_t)
        .mix(pgen_t)
        .mix(presub_t)
        .map { r -> tuple(r.method, r.csv) }
        .groupBy()

    memory_grouped = svar_m
        .mix(svar2_m)
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
    // Every task in the DAG must land on carter-cn-04: the campaign's Nextflow
    // work directory lives on /local/$USER (node-local scratch, not NFS -- the
    // heavy BENCH_PGEN_*/BUILD_*_FROM_PGEN processes already independently
    // rebuild a large PGEN random-access index per task, so keeping that I/O
    // off NFS matters). A task landing on any other node in the
    // `carter-compute` partition would stage into a work dir that doesn't
    // exist there and fail outright. This process previously had no pin
    // because it isn't a timed measurement -- but it still needs the pin for
    // this structural reason, not a measurement-purity one.
    clusterOptions '--nodelist=carter-cn-04'
    cpus 1
    time 30.min
    memory 32.GB

    input:
    svar: Path

    script:
    """
    make_sample_list.py ${svar} /dev/null --print-total
    """

    output:
    n: Integer = stdout().trim().toInteger()
}

process GENERATE_PAIRS {
    queue 'carter-compute'
    // See COUNT_FULL_SAMPLES: pinned so this task's work dir (on
    // /local/$USER, node-local to carter-cn-04) actually exists on whichever
    // node Slurm schedules it to.
    clusterOptions '--nodelist=carter-cn-04'
    cpus 2
    time 2.h
    memory 16.GB

    input:
    query_length: Integer
    n_replicates: Integer
    stream_batches: Integer
    seed: Integer
    bp_budget: Integer
    n_samples_full: Integer

    script:
    """
    generate_pairs.py \\
      ${params.svar} \\
      ${params.fai} \\
      ${query_length} \\
      pairs_${query_length}.parquet \\
      --seed ${seed} \\
      --n-replicates ${n_replicates} \\
      --stream-batches ${stream_batches} \\
      --bp-budget ${bp_budget}
    """

    output:
    record(
        query_length: query_length,
        n_samples: n_samples_full,
        pairs: file("pairs_${query_length}.parquet"),
    )
}

process MAKE_SAMPLE_LIST {
    queue 'carter-compute'
    // See COUNT_FULL_SAMPLES.
    clusterOptions '--nodelist=carter-cn-04'
    cpus 1
    time 30.min
    memory 32.GB

    input:
    n: Integer
    seed: Integer
    svar: Path

    script:
    """
    make_sample_list.py ${svar} samples_N${n}.txt --n ${n} --seed ${seed}
    """

    output:
    record(n: n, samples: file("samples_N${n}.txt"))
}

process SUBSET_BCF {
    queue 'carter-compute'
    // See COUNT_FULL_SAMPLES.
    clusterOptions '--nodelist=carter-cn-04'
    cpus 4
    time 4.h
    memory 16.GB

    input:
    n: Integer
    samples: Path
    bcf: Path

    script:
    """
    bcftools view -S ${samples} -c 1 --no-update --threads ${task.cpus} -W -Ob -o N${n}.bcf ${bcf}
    """

    output:
    record(n: n, bcf: file("N${n}.bcf"), csi: file("N${n}.bcf.csi"))
}

process SUBSET_PGEN {
    queue 'carter-compute'
    // See COUNT_FULL_SAMPLES.
    clusterOptions '--nodelist=carter-cn-04'
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

    // `vzs` writes the .pvar as Zstd-compressed (.pvar.zst) instead of plain
    // text -- ~31x smaller on this data (the source dir's own
    // 1kGP.snp_indel.split_multiallelics.pvar.zst is 2.6GB against the
    // 80.7GB plain .pvar). .pvar is per-VARIANT metadata, so it barely
    // shrinks as samples drop -- this was THE dominant driver of the
    // campaign's ENOSPC failure (N3202: 76GB .pvar; N1000: 62GB; N316:
    // 44GB, for local /local/$USER work-dir copies that used to coexist
    // uncompressed). Proven timing-neutral by reading genoray's PGEN
    // reader (_pgen.py): `.pvar`/`.pvar.zst` are referenced ONLY inside
    // `_index_path()`/`_load_index()`/`_write_index()`/`_scan_pvar()` --
    // i.e. only at index-*build* time (once per subset, off any timed
    // benchmark path) or as a same-cost `Path.exists()` check. Every read
    // method (`read()`, `read_ranges()`, `_read_genos()`, etc.) operates
    // exclusively on `self._index` (loaded once into memory from the
    // cached `.gvi`) and the `.pgen` binary via pgenlib -- the `.pvar`/
    // `.pvar.zst` text is never touched again. genoray already supports
    // `.pvar.zst` as a first-class fallback (`_index_path`'s explicit
    // `.pvar` -> `.pvar.zst` check; `_scan_pvar` opens `.zst` via
    // `ZstdFile`), so no downstream code needs to change -- SparseVar2's
    // own `_find_pvar` has the identical fallback, and `genoray write`'s
    // `PGEN(...)` construction (BUILD_SVAR_FROM_PGEN, below) uses the same
    // `_index_path()`.
    //
    // `pvar-cols=maybecm` (a --make-pgen MODIFIER, not a standalone flag --
    // `--pvar-cols=` alone is rejected as an unrecognized flag) drops the
    // optional xheader/QUAL/FILTER/INFO column sets, leaving the five
    // mandatory .pvar columns #CHROM/POS/ID/REF/ALT. CM is all-zero on this
    // data, so `maybecm` emits no CM column either. `vzs` alone was NOT
    // enough and this is why: the source .pvar's INFO column is a ~1kB
    // per-variant allele-frequency block (AC/AF/AN across five
    // superpopulations x rel/unrel), which is essentially the whole 80.7GB.
    // plink2 copies INFO through verbatim -- and its AC/AF are the ORIGINAL
    // full-cohort values, never recomputed for the subset, so they are stale
    // as well as unused. Worse, genoray's `_write_index` persists EVERY
    // column present in the .pvar into the Arrow-IPC `.gvi`, so compressing
    // the .pvar to 4.6GB still produced a ~66GB `N1000.pvar.zst.gvi`
    // (sink_ipc's per-row zstd recovers far less on INFO than plink2's
    // whole-stream zstd) -- and BUILD_SVAR_FROM_PGEN's `.svar/index.arrow`
    // is a second, byte-identical copy of that same index, so every cohort
    // point paid for the INFO index TWICE. That is what ENOSPC'd the
    // campaign twice.
    //
    // MEASURED equivalence on a 200-sample chr22 subset (DEF = current
    // flags, MIN = with `pvar-cols=maybecm`):
    //   .pgen bitwise identical; .psam identical; 270,568 variants both;
    //   CHROM/POS/ID/REF/ALT md5 identical; genoray's loaded index columns
    //   and their CHROM/POS/REF/ILEN values all equal (INFO never survives
    //   `_load_index`, which projects to
    //   ['index','CHROM','POS','REF','ALT','ILEN']).
    //   .pvar.zst 38.7MB -> 2.2MB (17.5x); `.gvi` 339MB -> 23.9MB (14.2x).
    //   SVAR2 store: all 17 files byte-identical.
    //   SVAR v1 read: genotype md5 identical, 85.14 -> 85.23 M calls/s
    //   (0.1%, noise), resident index 10,052,212 -> 10,052,218 bytes.
    //   Only `setup_ns` changes (123.8ms -> 24.8ms, a smaller index.arrow to
    //   load); it is recorded as its own column and is not the timed read.
    // ILEN is safe because this dataset has ZERO symbolic ALT alleles, so
    // `_load_index`'s INFO-present branch (regex SVLEN/END/IMPRECISE ->
    // `_symbolic_ilen()`) and its INFO-absent branch agree on every row --
    // verified by comparing ILEN values directly, not just hashes.
    script:
    """
    awk 'BEGIN{OFS="\\t"} {print "0", \$1}' ${samples} > keep.tsv
    plink2 --pfile in --keep keep.tsv --mac 1 --nonfounders --make-pgen vzs pvar-cols=maybecm --threads ${task.cpus} --out N${n}
    """

    output:
    record(
        n: n,
        pgen: file("N${n}.pgen"),
        pvar: file("N${n}.pvar.zst"),
        psam: file("N${n}.psam"),
    )
}

process BUILD_SVAR_FROM_PGEN {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 4
    time 8.h
    memory 128.GB

    input:
    n: Integer
    pgen: Path
    pvar: Path
    psam: Path

    // `genoray write` constructs a `PGEN(pgen)` reader internally (same
    // class/index cache as bench_pgen.py and the CLI's `write` command,
    // confirmed by reading genoray/_cli/__main__.py), which unconditionally
    // builds a `.gvi` index as a side effect if a valid one doesn't already
    // exist next to `pvar` -- there never is one here, since `pvar` is a
    // freshly-generated subset every run. Declaring that `.gvi` as a named
    // output (instead of leaving it an untracked side effect) lets
    // downstream consumers of this n's subset (BENCH_PGEN_THROUGHPUT/
    // BENCH_PGEN_MEMORY, via SubsetStores.pvar_gvi / SweepInput.gvi) reuse
    // it instead of each independently rebuilding their own copy -- before
    // this, 3 tasks per cohort-sweep n-value each built their own copy of
    // an index that was 20GB+ even for the smallest n=32 subset (measured
    // from the failed campaign's disk footprint), a second, independent
    // driver of the ENOSPC failure alongside SUBSET_PGEN's .pvar text.
    // `.pvar.zst.gvi`, not `.pvar.gvi`: genoray's index path is derived
    // from whichever of `.pvar`/`.pvar.zst` it actually finds next to
    // `pgen` (see SUBSET_PGEN above, which now only ever produces
    // `.pvar.zst`), and it appends `.gvi` to that file's own suffix --
    // verified directly: `Path("N10.pvar.zst").with_suffix(".zst.gvi")` ==
    // `Path("N10.pvar.zst.gvi")`.
    script:
    """
    genoray write ${pgen} N${n}.svar --threads ${task.cpus}
    """

    output:
    record(n: n, svar: file("N${n}.svar"), gvi: file("N${n}.pvar.zst.gvi"))
}

process BUILD_SVAR2_FROM_PGEN {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 4
    time 8.h
    memory 128.GB

    // beforeScript that activates the `svar2` pixi env lives in
    // nextflow.config's `withName: 'BENCH_SVAR2_.*|BUILD_SVAR2_.*'` selector.

    input:
    n: Integer
    pgen: Path
    pvar: Path
    psam: Path

    script:
    """
    build_svar2_from_pgen.py ${pgen} N${n}.svar2 --threads ${task.cpus}
    """

    output:
    record(n: n, svar2: file("N${n}.svar2"))
}

process GENERATE_PAIRS_N {
    queue 'carter-compute'
    // See COUNT_FULL_SAMPLES.
    clusterOptions '--nodelist=carter-cn-04'
    cpus 2
    time 2.h
    memory 16.GB

    input:
    t: SubsetStores
    query_length: Integer
    n_replicates: Integer
    stream_batches: Integer
    seed: Integer
    bp_budget: Integer

    script:
    """
    generate_pairs.py \\
      ${t.svar} \\
      ${params.fai} \\
      ${query_length} \\
      pairs_N${t.n}.parquet \\
      --seed ${seed} \\
      --n-replicates ${n_replicates} \\
      --stream-batches ${stream_batches} \\
      --bp-budget ${bp_budget}
    """

    output:
    record(
        query_length: query_length,
        n_samples: t.n,
        pairs: file("pairs_N${t.n}.parquet"),
        svar: t.svar,
        svar2: t.svar2,
        bcf: t.bcf,
        bcf_csi: t.bcf_csi,
        pgen: t.pgen,
        pvar: t.pvar,
        psam: t.psam,
        gvi: t.pvar_gvi,
    )
}

process BENCH_SVAR_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

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
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
    """

    output:
    record(method: "svar", csv: file("svar_q${p.query_length}_n${p.n_samples}_throughput.csv"))
}

process BENCH_SVAR_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

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
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
    """

    output:
    record(method: "svar", csv: file("svar_q${p.query_length}_n${p.n_samples}_memory.csv"))
}

process BENCH_SVAR2_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    // beforeScript that activates the `svar2` pixi env lives in
    // nextflow.config's `withName: 'BENCH_SVAR2_.*|BUILD_SVAR2_.*'` selector.

    input:
    p: SweepInput

    script:
    """
    bench_svar2.py \\
      ${p.pairs} \\
      ${p.svar2} \\
      svar2_q${p.query_length}_n${p.n_samples}_throughput.csv \\
      --dataset ${params.dataset} \\
      --mode throughput \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
    """

    output:
    record(method: "svar2", csv: file("svar2_q${p.query_length}_n${p.n_samples}_throughput.csv"))
}

process BENCH_SVAR2_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    // beforeScript that activates the `svar2` pixi env lives in
    // nextflow.config's `withName: 'BENCH_SVAR2_.*|BUILD_SVAR2_.*'` selector.

    input:
    p: SweepInput

    script:
    """
    bench_svar2.py \\
      ${p.pairs} \\
      ${p.svar2} \\
      svar2_q${p.query_length}_n${p.n_samples}_memory.csv \\
      --dataset ${params.dataset} \\
      --mode memory \\
      --n-samples ${p.n_samples} \\
      --min-seconds ${params.min_seconds} \\
      --min-batches ${params.min_batches}
    """

    output:
    record(method: "svar2", csv: file("svar2_q${p.query_length}_n${p.n_samples}_memory.csv"))
}

process BENCH_BCF_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

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

    output:
    record(method: "bcf", csv: file("bcf_q${p.query_length}_n${p.n_samples}_throughput.csv"))
}

process BENCH_BCF_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

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

    output:
    record(method: "bcf", csv: file("bcf_q${p.query_length}_n${p.n_samples}_memory.csv"))
}

process BENCH_PGEN_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    // Was 1.d. query_length=2048 is single-pass over 65536 pairs/replicate
    // x 5 replicates in production (verified against the pre-SVAR2
    // committed svar_throughput.csv, which independently shows n_pairs=
    // 65536 at query_length=2048). Measured PGEN per-pair cost in the smoke
    // run: ~0.20-0.23s/pair -> ~19.1h for that read loop alone at
    // query_length=2048, even with the index-rebuild tax removed (staging
    // the existing .gvi below saves ~28-29min out of that, not the
    // dominant term). 36h gives ~1.9x headroom over the ~19.1-19.6h
    // estimate, comfortably under the partition's 14-day cap.
    time 36.h
    memory 64.GB

    input:
    // p.gvi is staged (basename-matched) alongside p.pgen/p.pvar/p.psam so
    // genoray finds a valid index instead of rebuilding one: the full
    // cohort's pre-built index for primary-sweep records, or that n's
    // once-built subset index (from BUILD_SVAR_FROM_PGEN) for cohort-sweep
    // records. See pvar_gvi_path's definition above.
    p: SweepInput

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

    output:
    record(method: "pgen", csv: file("pgen_q${p.query_length}_n${p.n_samples}_throughput.csv"))
}

process BENCH_PGEN_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    // See BENCH_PGEN_THROUGHPUT: p.gvi is staged so genoray finds an
    // already-built index instead of rebuilding one.
    p: SweepInput

    script:
    // q=2048 yields the largest batch (65536 pairs/rep). pgen reads run ~3/s, so
    // that one memory cell takes ~29h and overruns the 1d limit. pgen peak RSS is
    // dominated by the fixed ~20GB .gvi index load and plateaus (~24GB) regardless
    // of query count -- verified empirically: 512 vs 4096 pairs/rep both ~21-24GB,
    // AveRSS~MaxRSS -- so capping this cell does not bias its measurement.
    // NOTE: this cap is pgen-specific. Do NOT add it to BENCH_SVAR_MEMORY: SparseVar
    // is a memmap format, so its RSS = resident .svar pages and *legitimately* scales
    // with query count (10GB@1024 pairs -> 91GB@65536) -- that scaling is the real
    // behavior the memory benchmark measures. bcf/presubset stream at <1GB and finish
    // at full pairs, so they need no cap either. Other cells keep the original command
    // verbatim to preserve the resume cache.
    if( p.query_length <= 2048 )
        """
        bench_pgen.py \\
          ${p.pairs} \\
          ${p.pgen} \\
          pgen_q${p.query_length}_n${p.n_samples}_memory.csv \\
          --dataset ${params.dataset} \\
          --mode memory \\
          --n-samples ${p.n_samples} \\
          --max-pairs-per-rep 512
        """
    else
        """
        bench_pgen.py \\
          ${p.pairs} \\
          ${p.pgen} \\
          pgen_q${p.query_length}_n${p.n_samples}_memory.csv \\
          --dataset ${params.dataset} \\
          --mode memory \\
          --n-samples ${p.n_samples}
        """

    output:
    record(method: "pgen", csv: file("pgen_q${p.query_length}_n${p.n_samples}_memory.csv"))
}

process BENCH_PRESUBSET_BCF_THROUGHPUT {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

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

    output:
    record(method: "presubset_bcf", csv: file("presubset_bcf_q${p.query_length}_n${p.n_samples}_throughput.csv"))
}

process BENCH_PRESUBSET_BCF_MEMORY {
    queue 'carter-compute'
    clusterOptions '--nodelist=carter-cn-04'
    cpus 8
    time 1.d
    memory 64.GB

    input:
    p: SweepInput

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

    output:
    record(method: "presubset_bcf", csv: file("presubset_bcf_q${p.query_length}_n${p.n_samples}_memory.csv"))
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

record SubsetStores {
    n: Integer
    svar: Path
    svar2: Path
    bcf: Path
    bcf_csi: Path
    pgen: Path
    pvar: Path
    psam: Path
    // This n's genoray PGEN index, built once by BUILD_SVAR_FROM_PGEN and
    // reused by BENCH_PGEN_THROUGHPUT/BENCH_PGEN_MEMORY via SweepInput.gvi
    // -- see pvar_gvi_path's definition in the workflow block.
    pvar_gvi: Path
}

record SweepInput {
    query_length: Integer
    n_samples: Integer
    pairs: Path
    svar: Path
    svar2: Path
    bcf: Path
    bcf_csi: Path
    pgen: Path
    pvar: Path
    psam: Path
    // Genoray PGEN index for `pgen`/`pvar`: the full cohort's pre-built one
    // for primary-sweep records, or that n's once-built subset one for
    // cohort-sweep records. Staged into BENCH_PGEN_THROUGHPUT/
    // BENCH_PGEN_MEMORY alongside pgen/pvar/psam so genoray finds it valid
    // instead of rebuilding it.
    gvi: Path
}

record MethodResult {
    method: String
    csv: Path
}
