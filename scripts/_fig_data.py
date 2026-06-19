"""Shared data-loading helpers + constants for the GVL 0.27 throughput figures.

Imported by scripts/plot.py (standalone panels) and scripts/plot_figure2.py
(the unified Figure 2 composite). All throughput numbers come from the GVL
0.27.0 eager (mode=none) bench in results_gvl027/; see root CLAUDE.md.
"""

import glob
from pathlib import Path

import polars as pl

proj_dir = Path(__file__).resolve().parent.parent

RAM_BW_GBPS = 35.0
RAM_BW_LABEL = "cn-03 max RAM\nbandwidth"
GVL027_LABELS = {
    "TCGA_ATAC": "GVL: TCGA BRCA ATAC (n=62)",
    "1KGP": "GVL: 1000 Genomes (n=3,202)",
    "UKBB": "GVL: Biobank (n=487,409)",
}
# Variant file-type labels (mirrors variant_throughput/bin/_plot_common.py).
METHOD_LABELS = {
    "svar": "SVAR",
    "bcf": "BCF",
    "pgen": "PGEN",
    "presubset_bcf": "PRESUB-BCF",
}


def gvl027_peak(result_glob: str) -> pl.DataFrame:
    """Max eager (mode=none) throughput per (dataset, seqlen), in GB/s."""
    files = sorted(glob.glob(str(proj_dir / result_glob)))
    if not files:
        raise FileNotFoundError(f"no GVL 0.27.0 result CSVs matched {result_glob!r}")
    df = pl.concat([pl.read_csv(c) for c in files], how="vertical_relaxed").with_columns(
        pl.col("throughput (MiB/s)").cast(pl.Float64, strict=False)
    )
    df = df.filter(
        pl.col("throughput (MiB/s)").is_finite() & (pl.col("throughput (MiB/s)") > 0)
    )
    return (
        df.group_by("dataset", "seqlen")
        .agg(throughput=(pl.col("throughput (MiB/s)").max() * 2**20 / 1e9))
        .sort("dataset", "seqlen")
    )


def gvl027_grid(result_glob: str) -> pl.DataFrame:
    """Per-cell eager (mode=none) throughput grid, GB/s, with n_nucleotides."""
    files = sorted(glob.glob(str(proj_dir / result_glob)))
    if not files:
        raise FileNotFoundError(f"no GVL 0.27.0 result CSVs matched {result_glob!r}")
    df = pl.concat([pl.read_csv(c) for c in files], how="vertical_relaxed").with_columns(
        pl.col("throughput (MiB/s)").cast(pl.Float64, strict=False)
    )
    df = df.filter(
        pl.col("throughput (MiB/s)").is_finite() & (pl.col("throughput (MiB/s)") > 0)
    )
    return df.with_columns(
        n_nucleotides=pl.col("seqlen") * pl.col("batch_size"),
        throughput=pl.col("throughput (MiB/s)") * 2**20 / 1e9,  # GB/s
    )


def disk_usage_df() -> pl.DataFrame:
    """Personalized-genome disk footprint, GVL vs compressed FASTA (Fig. 2A)."""
    compressed_hg37 = 0.987
    # GDC (n=16,007) omitted: it appears in no other benchmark, so showing it
    # only here would draw reviewer questions. (compressed_hg38 was only used for it.)
    return pl.DataFrame({
        "Dataset": [
            "TCGA BRCA ATAC (n=62)",
            "TCGA BRCA ATAC (n=62)",
            "1000 Genomes (n=3,202)",
            "1000 Genomes (n=3,202)",
            "Biobank, chr22 (n=487,409)",
            "Biobank, chr22 (n=487,409)",
        ],
        "Implementation": ["GVL", "FASTA"] * 3,
        "Disk Space (GB)": [
            0.173,
            compressed_hg37 * 62 * 2,
            3.1,
            compressed_hg37 * 3202 * 2,
            30,
            0.0096 * 487409 * 2,  # just chr22
        ],
    })


def variant_qlen_df() -> pl.DataFrame:
    """Variant calls/sec vs query length, by file type (Fig. 2B), at full cohort.

    Reads variant_throughput/results/*_throughput.csv, restricted to the largest
    cohort (n_samples == max), matching the query-length view in
    variant_throughput/bin/plot_throughput.py. Returns columns:
    log10_query_length, log10_calls_per_sec, method_label, calls_per_sec.
    """
    csvs = sorted(glob.glob(str(proj_dir / "variant_throughput/results/*_throughput.csv")))
    if not csvs:
        raise FileNotFoundError("no variant_throughput/results/*_throughput.csv found")
    raw = pl.concat(
        [pl.read_csv(p, schema_overrides={"setup_ns": pl.Int64}) for p in csvs],
        how="vertical_relaxed",
    )
    full_n = raw["n_samples"].max()
    return (
        raw.filter(pl.col("n_samples") == full_n)
        .filter((pl.col("n_calls") > 0) & (pl.col("elapsed_ns") > 0))
        .with_columns(
            (pl.col("n_calls") / (pl.col("elapsed_ns") * 1e-9)).alias("calls_per_sec")
        )
        .with_columns(
            pl.col("query_length").log(base=10).alias("log10_query_length"),
            pl.col("calls_per_sec").log(base=10).alias("log10_calls_per_sec"),
            pl.col("method").replace(METHOD_LABELS).alias("method_label"),
        )
    )
