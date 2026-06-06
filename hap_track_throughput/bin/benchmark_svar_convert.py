#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def bench(
    results: Path,
    variants: Path,
    out_svar: Path,
    dataset: str = "",
    measure_memory: bool = False,
    max_mem: str = "4g",
    n_jobs: int = -1,
    no_symbolic: bool = True,
    no_breakend: bool = True,
):
    from time import perf_counter_ns

    import genoray

    from _genoray_filter import open_filtered_reader

    source, source_fmt = open_filtered_reader(variants, no_symbolic, no_breakend)

    if source_fmt == "pgen":

        def convert():
            genoray.SparseVar.from_pgen(out_svar, source, max_mem, overwrite=True, n_jobs=n_jobs)

    else:  # bcf / vcf

        def convert():
            genoray.SparseVar.from_vcf(out_svar, source, max_mem, overwrite=True, n_jobs=n_jobs)

    dataset = dataset or variants.stem

    if measure_memory:
        from _mem_sampler import PeakRssSampler

        with PeakRssSampler() as s:
            convert()
        with open(results, "w") as f:
            f.write("dataset,source_fmt,n_jobs,avg_rss_bytes,peak_rss_bytes\n")
            f.write(f"{dataset},{source_fmt},{n_jobs},{s.avg},{s.peak}\n")
    else:
        t0 = perf_counter_ns()
        convert()
        duration = perf_counter_ns() - t0
        with open(results, "w") as f:
            f.write("dataset,source_fmt,n_jobs,duration\n")
            f.write(f"{dataset},{source_fmt},{n_jobs},{duration}\n")


if __name__ == "__main__":
    run(bench)
