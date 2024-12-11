# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

data_dir = Path(__file__).parent.parent / "throughput"
fig_dir = Path(__file__).parent
fig_dir.mkdir(parents=True, exist_ok=True)

# %% hap data
results = pl.read_csv(data_dir / "hap_results.csv").with_columns(
    n_nucleotides=pl.col("seqlen") * pl.col("batch_size"),
    throughput=pl.col("throughput (MiB/s)") * 2**20 / 1e9,  # GB/s
)
ref_results = (
    pl.read_csv(data_dir / "ref_results.csv")
    .with_columns(
        n_nucleotides=pl.col("seqlen") * pl.col("batch_size"),
        throughput=pl.col("throughput (MiB/s)") * 2**20 / 1e9,  # GB/s
    )
    .group_by("seqlen")
    .agg(pl.col("throughput").max())
)
results = results.join(
    ref_results.select("seqlen", ref_throughput="throughput"), on=["seqlen"]
).with_columns(
    pl.col("dataset").replace(
        {
            "tcga-atac": "GVL: TCGA BRCA ATAC (n=62)",
            "1kgp": "GVL: 1000 Genomes (n=3,202)",
            "ukbb": "GVL: Biobank (n=487,409)",
        }
    ),
    batch_mb=pl.col("batch_size") * pl.col("seqlen") / 1e6,
)

# %% track data
track_results = pl.read_csv(data_dir / "track_results.csv").with_columns(
    n_nucleotides=pl.col("seqlen") * pl.col("batch_size"),
    batch_mb=pl.col("batch_size") * pl.col("seqlen") * 4 / 1e6,
    throughput=pl.col("throughput (MiB/s)") * 2**20 / 1e9,  # GB/s
)
pybigwig_results = (
    pl.read_csv(data_dir / "pybigwig_results.csv")
    .with_columns(
        n_nucleotides=pl.col("seqlen") * pl.col("batch_size"),
        batch_mb=pl.col("batch_size") * pl.col("seqlen") * 4 / 1e6,
        throughput=pl.col("throughput (MiB/s)") * 2**20 / 1e9,  # GB/s
    )
    .group_by("seqlen")
    .agg(pl.col("throughput").max())
)
track_results = track_results.join(
    pybigwig_results.select("seqlen", bigwig_throughput="throughput"), on="seqlen"
)

# %% track perf_ratio best perf
track_perf_ratio = (
    track_results.group_by("dataset", "seqlen")
    .agg(
        pl.col("throughput").max(),
        perf_ratio=pl.col("throughput").max() / pl.col("bigwig_throughput").max(),
    )
    .sort("perf_ratio")
)
# %%
# best track results
with sns.plotting_context("notebook", font_scale=1.3):
    fig, ax = plt.subplots(dpi=150)
    sns.lineplot(
        data=track_perf_ratio.to_pandas(),
        x="seqlen",
        y="throughput",
        ax=ax,
        label="GVL",
        linewidth=5,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    sns.lineplot(
        data=pybigwig_results.to_pandas(),
        x="seqlen",
        y="throughput",
        ax=ax,
        color="C2",
        label="BigWig",
        linewidth=5,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    ax.axhline(31.5, c="k", ls="--", alpha=0.5, linewidth=5)
    ax.text(
        pybigwig_results["seqlen"].min() - 1000,
        31.5,
        r"A100 CPU$\rightarrow$GPU" + "\ntransfer limit",
        va="center",
        ha="right",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Sequence length",
        ylabel="Throughput (GB/s)",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "best_track_performance.png", dpi=300)
    plt.savefig(fig_dir / "best_track_performance.pdf")

# %% perf_ratio
perf_ratio = (
    results.group_by("dataset", "seqlen")
    .agg(
        pl.col("throughput").max(),
        perf_ratio=pl.col("throughput").max() / pl.col("ref_throughput").max(),
    )
    .sort("perf_ratio", descending=True)
)

# %% best haplotype performance
with sns.plotting_context("notebook", font_scale=1.3):
    fig, ax = plt.subplots(dpi=150)
    sns.lineplot(
        data=perf_ratio.to_pandas(),
        x="seqlen",
        y="throughput",
        hue="dataset",
        hue_order=[
            "GVL: TCGA BRCA ATAC (n=62)",
            "GVL: 1000 Genomes (n=3,202)",
            "GVL: Biobank (n=487,409)",
        ],
        ax=ax,
        linewidth=4,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    sns.lineplot(
        data=ref_results.to_pandas(),
        x="seqlen",
        y="throughput",
        ax=ax,
        color="C3",
        label="FASTA",
        linewidth=4,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    ax.axhline(31.5, c="k", ls="--", alpha=0.5, linewidth=5)
    ax.text(
        ref_results["seqlen"].min() - 1000,
        31.5,
        r"A100 CPU$\rightarrow$GPU" + "\ntransfer limit",
        va="center",
        ha="right",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Sequence length",
        ylabel="Throughput (GB/s)",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "best_haplotype_performance.png", dpi=300)
    plt.savefig(fig_dir / "best_haplotype_performance.pdf")

# %% disk usage
with sns.plotting_context("notebook", font_scale=1.4):
    fig, ax = plt.subplots(figsize=(7, 4))
    compressed_hg37 = 0.987
    memory = pl.DataFrame(
        {
            "Dataset": [
                "TCGA BRCA ATAC (n=62)",
                "TCGA BRCA ATAC (n=62)",
                "1000 Genomes (n=3,202)",
                "1000 Genomes (n=3,202)",
                "Biobank, chr22 (n=487,409)",
                "Biobank, chr22 (n=487,409)",
            ],
            "Implementation": ["GVL", "FASTA", "GVL", "FASTA", "GVL", "FASTA"],
            "Disk Space (GB)": [
                0.173,
                compressed_hg37 * 62 * 2,
                3.1,
                compressed_hg37 * 3202 * 2,
                30,
                0.0096 * 487409 * 2,
            ],
        }
    )
    ax = sns.barplot(
        data=memory.to_pandas(),
        x="Disk Space (GB)",
        y="Dataset",
        hue="Implementation",
        ax=ax,
    )
    ax.set(xscale="log")
    fig.tight_layout()
    fig.savefig(fig_dir / "disk_usage.png", dpi=300)
    fig.savefig(fig_dir / "disk_usage.pdf")
