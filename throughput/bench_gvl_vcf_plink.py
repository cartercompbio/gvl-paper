#! /usr/bin/env python


from pathlib import Path


def main(
    svar: Path,
    bcf: Path,
    pgen: Path,
    fai: Path,
    repeats: int,
    query_lengths: int | list[int],
    results: Path,
    seed: int = 0,
):
    import random
    from time import perf_counter_ns

    import awkward as ak
    import polars as pl
    from awkward.contents import ListArray, NumpyArray, RegularArray
    from awkward.index import Index
    from genoray import PGEN, VCF, SparseVar
    from rich.progress import MofNCompleteColumn, Progress

    if isinstance(query_lengths, int):
        query_lengths = [query_lengths]

    _svar = SparseVar(svar)
    _bcf = VCF(bcf, with_gvi_index=False)
    _pgen = PGEN(pgen)

    _fai = (
        pl.read_csv(
            fai,
            separator="\t",
            has_header=False,
            new_columns=["contig", "length"],
        )
        .select(
            pl.col("contig").replace_strict(_svar._c_norm.contig_map, default=None),
            "length",
        )
        .drop_nulls()
        .filter(pl.col("contig").is_in(_svar.contigs))
    )
    contig_lengths: dict[str, int] = {
        c: l for c, l in zip(*_fai.get_columns()) if c in _svar.contigs
    }
    contigs = list(contig_lengths)

    random.seed(seed)
    rec_q_len = []
    rec_n_calls = []
    rec_n_variants = []
    svar_times = []
    bcf_times = []
    plink_times = []

    pbar = Progress(*Progress.get_default_columns(), MofNCompleteColumn())
    pbar.start()

    n_queries = repeats * len(query_lengths)
    task = pbar.add_task("Benchmarking", total=n_queries)
    for q_len in query_lengths:
        for _ in range(repeats):
            rec_q_len.append(q_len)

            contig = random.choice(contigs)
            start = random.randint(
                max(0, contig_lengths[contig] // 8),
                min(contig_lengths[contig] - q_len, 3 * contig_lengths[contig] // 8),
            )

            end = start + q_len
            sample = random.choice(_svar.available_samples)

            # (2, ploidy)
            offsets = _svar._find_starts_ends(contig, start, end, sample).squeeze()
            n_calls = (offsets[1] - offsets[0]).sum()
            rec_n_calls.append(n_calls)

            node = NumpyArray(_svar.genos.data)  # type: ignore
            node = ListArray(Index(offsets[0].ravel()), Index(offsets[1].ravel()), node)
            node = RegularArray(node, 2)

            t0 = perf_counter_ns()
            ak.to_packed(node)
            t1 = perf_counter_ns()
            svar_times.append(t1 - t0)

            t0 = perf_counter_ns()
            _bcf = _bcf.set_samples(sample)
            # (s p v)
            genos = _bcf.read(contig, start, end, mode=_bcf.Genos8)
            n_variants = genos.shape[-1]
            t1 = perf_counter_ns()
            bcf_times.append(t1 - t0)
            rec_n_variants.append(n_variants)

            t0 = perf_counter_ns()
            _pgen = _pgen.set_samples(sample)
            _pgen.read(contig, start, end, mode=_pgen.Genos)
            t1 = perf_counter_ns()
            plink_times.append(t1 - t0)

            pbar.update(task, advance=1)

    pbar.stop()

    pl.DataFrame(
        {
            "query_length": rec_q_len,
            "n_calls": rec_n_calls,
            "n_variants": rec_n_variants,
            "svar_time": svar_times,
            "bcf_time": bcf_times,
            "plink_time": plink_times,
        }
    ).write_csv(results)


if __name__ == "__main__":
    # cyclopts.run(main)

    import numpy as np

    repeats = 5
    query_lengths = (2 ** np.arange(11, 25)).tolist()
    res_dir = Path("/cellar/users/dlaub/projects/gvl-paper/results")

    # ddir = Path("/cellar/users/dlaub/data/1kGP")
    # main(
    #     svar=ddir / "1kGP.snp_indel.split_multiallelics.svar/",
    #     bcf=ddir / "1kGP.snp_indel.split_multiallelics.bcf",
    #     pgen=ddir / "1kGP.snp_indel.split_multiallelics.pgen",
    #     fai=ddir / "GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    #     repeats=repeats,
    #     query_lengths=query_lengths,
    #     results=res_dir / "variants_random_read_throughput_1kgp.csv",
    # )

    ddir = Path("/carter/shared/data/gdc/somatic/wgs_DR45/results")
    main(
        svar=ddir / "gdc_wgs_DR45.svar/",
        bcf=ddir / "gdc_wgs_DR45.bcf",
        pgen=ddir / "gdc_wgs_DR45.gt.pgen",
        fai=Path("/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa.fai"),
        repeats=repeats,
        query_lengths=query_lengths,
        results=res_dir / "variants_random_read_throughput_gdc.csv",
    )
