#! /usr/bin/env python3

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

sns.set_context("notebook", font_scale=1.5)
proj_dir = Path(__file__).parent.parent
data_dir = proj_dir / "results"
fig_dir = proj_dir / "figures"
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

fg = sns.relplot(
    results.unpivot(
        ["throughput", "ref_throughput"],
        index=["seqlen", "threads", "dataset", "batch_size", "n_nucleotides"],
        variable_name="filetype",
        value_name="throughput",
    )
    .with_columns(
        pl.col("filetype")
        .replace_strict({"throughput": "GVL", "ref_throughput": "FASTA"})
        .alias("File type"),
    )
    .rename({"threads": "Threads"}),
    x="n_nucleotides",
    y="throughput",
    hue="Threads",
    style="File type",
    kind="line",
    markers=True,
    legend="full",
    aspect=1.5,
    markersize=10,
)
fg.set(
    xlabel="Nucleotides per batch",
    xscale="log",
    ylabel="Throughput (GB/s)",
    yscale="log",
)
ax = fg.axes[0, 0]
ax.axhline(
    31.5,
    c="k",
    ls="--",
    alpha=0.5,
    linewidth=3,
)
ax.text(
    results["n_nucleotides"].min() - 1500,  # pyright: ignore
    31.5,
    r"A100 CPU$\rightarrow$GPU" + "\ntransfer limit",
    va="center",
    ha="right",
)
fg.tight_layout()
fg.savefig(fig_dir / "hap_throughput.svg")
fg.savefig(fig_dir / "hap_throughput.png", dpi=150)

# %%
fg = sns.relplot(
    track_results.unpivot(
        ["throughput", "bigwig_throughput"],
        index=["seqlen", "threads", "n_nucleotides"],
        variable_name="filetype",
        value_name="throughput",
    )
    .with_columns(
        pl.col("filetype")
        .replace_strict({"throughput": "GVL", "bigwig_throughput": "pyBigWig"})
        .alias("File type"),
    )
    .rename({"threads": "Threads"}),
    x="n_nucleotides",
    y="throughput",
    hue="Threads",
    style="File type",
    kind="line",
    markers=True,
    legend="full",
    aspect=1.5,
    markersize=10,
)
fg.set(
    xlabel="Track values per batch",
    xscale="log",
    ylabel="Throughput (GB/s)",
    yscale="log",
)
ax = fg.axes[0, 0]
ax.axhline(
    31.5,
    c="k",
    ls="--",
    alpha=0.5,
    linewidth=3,
)
ax.text(
    track_results["n_nucleotides"].min() - 3000,  # pyright: ignore
    31.5,
    r"A100 CPU$\rightarrow$GPU" + "\ntransfer limit",
    va="center",
    ha="right",
)
fg.tight_layout()
fg.savefig(fig_dir / "track_throughput.svg")
fg.savefig(fig_dir / "track_throughput.png", dpi=150)

# %%
# best track results
fig, ax = plt.subplots()
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
    pybigwig_results["seqlen"].min() - 1000,  # pyright: ignore
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
plt.savefig(fig_dir / "best_track_performance.svg")

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
data = pl.concat(
    [perf_ratio.drop("perf_ratio"), ref_results.with_columns(dataset=pl.lit("FASTA"))],
    how="diagonal_relaxed",
).rename({"dataset": "Dataset"})

fg = sns.relplot(
    data,
    x="seqlen",
    y="throughput",
    hue="Dataset",
    hue_order=[
        "GVL: TCGA BRCA ATAC (n=62)",
        "GVL: 1000 Genomes (n=3,202)",
        "GVL: Biobank (n=487,409)",
        "FASTA",
    ],
    linewidth=4,
    solid_joinstyle="round",
    solid_capstyle="round",
    kind="line",
    aspect=0.6,
)
ax = fg.ax
ax.axhline(31.5, c="k", ls="--", alpha=0.5, linewidth=3)
ax.text(
    ref_results["seqlen"].min() - 1000,  # pyright: ignore
    40,
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
plt.savefig(fig_dir / "best_haplotype_performance.svg")

# %% disk usage
compressed_hg37 = 0.987
compressed_hg38 = 0.875
memory = pl.DataFrame(
    {
        "Dataset": [
            "TCGA BRCA ATAC (n=62)",
            "TCGA BRCA ATAC (n=62)",
            "1000 Genomes (n=3,202)",
            "1000 Genomes (n=3,202)",
            "GDC (n=16,007)",
            "GDC (n=16,007)",
            "Biobank, chr22 (n=487,409)",
            "Biobank, chr22 (n=487,409)",
        ],
        "Implementation": ["GVL", "FASTA"] * 4,
        "Disk Space (GB)": [
            0.173,
            compressed_hg37 * 62 * 2,
            3.1,
            compressed_hg37 * 3202 * 2,
            7.9,
            compressed_hg38 * 16007,
            30,
            0.0096 * 487409 * 2,  # just chr22
        ],
    }
)
fg = sns.catplot(
    memory,
    x="Disk Space (GB)",
    y="Dataset",
    hue="Implementation",
    kind="bar",
    aspect=1.5,
)
fg.set(xscale="log")
fg.tight_layout()
fg.savefig(fig_dir / "disk_usage.png", dpi=300)
fg.savefig(fig_dir / "disk_usage.pdf")
fg.savefig(fig_dir / "disk_usage.svg")


# %%
var_throughput = (
    pl.read_csv(proj_dir / "results" / "variants_random_read_throughput_1kgp.csv")
    .unpivot(
        ["svar_time", "bcf_time", "plink_time"],
        index=["query_length", "n_calls", "n_variants"],
        variable_name="filetype",
        value_name="time",
    )
    .with_columns(
        pl.col("filetype")
        .str.split("_")
        .list.get(0)
        .str.to_uppercase()
        .replace({"PLINK": "PGEN"})
    )
)

q_len_name = r"$\log_{10}$ query length"
print(
    var_throughput.filter(pl.col("filetype") == "SVAR")
    .drop("filetype")
    .join(
        var_throughput.filter(pl.col("filetype") == "BCF").drop("filetype"),
        ["query_length", "n_variants", "n_calls"],
        suffix="_bcf",
    )
    .join(
        var_throughput.filter(pl.col("filetype") == "PGEN").drop("filetype"),
        ["query_length", "n_variants", "n_calls"],
        suffix="_pgen",
    )
    .with_columns(
        ratio_bcf=pl.col("time_bcf") / pl.col("time"),
        ratio_pgen=pl.col("time_pgen") / pl.col("time"),
    )
    .group_by("query_length")
    .agg(
        pl.col("ratio_bcf").mean(),
        pl.col("ratio_pgen").mean(),
    )
    .max()
)

data = var_throughput.with_columns(
    pl.col("query_length").log(10),
    log_vars_per_sec=(pl.col("n_variants") / pl.col("time") * 1e9).log(10),
).rename({"query_length": q_len_name, "filetype": "File type"})
fg = sns.lmplot(
    data.to_pandas(),
    x=q_len_name,
    y="log_vars_per_sec",
    hue="File type",
    aspect=0.6,
    lowess=True,
)
_ = fg.set(xlabel=q_len_name, ylabel=r"$\log_{10}$ variants/s")
sns.move_legend(fg, "center left", bbox_to_anchor=(0.95, 0.5))
fg.figure.tight_layout()
fg.savefig(
    fig_dir / "variant_throughput.png",
    dpi=300,
    bbox_inches="tight",
)
fg.savefig(fig_dir / "variant_throughput.svg")
