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
):
    from time import perf_counter_ns

    import genoray

    name = variants.name.lower()
    if name.endswith(".pgen"):
        source = genoray.PGEN(variants)
        source_fmt = "pgen"

        def convert():
            genoray.SparseVar.from_pgen(out_svar, source, max_mem, overwrite=True, n_jobs=n_jobs)

    elif name.endswith(".bcf") or name.endswith(".vcf") or name.endswith(".vcf.gz"):
        source = genoray.VCF(variants)
        source_fmt = "bcf" if name.endswith(".bcf") else "vcf"

        def convert():
            genoray.SparseVar.from_vcf(out_svar, source, max_mem, overwrite=True, n_jobs=n_jobs)

    else:
        raise ValueError(f"Unsupported variant format: {variants}")

    dataset = dataset or variants.stem

    if measure_memory:
        from _mem_sampler import PeakRssSampler

        with PeakRssSampler() as s:
            convert()
        with open(results, "w") as f:
            f.write("dataset,source_fmt,n_jobs,peak_rss_bytes\n")
            f.write(f"{dataset},{source_fmt},{n_jobs},{s.peak}\n")
    else:
        t0 = perf_counter_ns()
        convert()
        duration = perf_counter_ns() - t0
        with open(results, "w") as f:
            f.write("dataset,source_fmt,n_jobs,duration\n")
            f.write(f"{dataset},{source_fmt},{n_jobs},{duration}\n")


if __name__ == "__main__":
    run(bench)
