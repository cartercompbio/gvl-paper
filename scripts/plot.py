#! /usr/bin/env python3

# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_data import (  # noqa: E402
    GVL027_LABELS,
    RAM_BW_GBPS,
    RAM_BW_LABEL,
    disk_usage_df,
    gvl027_grid,
    gvl027_peak,
)

sns.set_context("notebook", font_scale=1.5)
proj_dir = Path(__file__).parent.parent
data_dir = proj_dir / "results"
fig_dir = proj_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ALL throughput figures are now on the GVL 0.27.0 eager (mode=none) full bench in
# results_gvl027/ (artifact-free per-batch decode; see CLAUDE.md): the Fig 2C/2D
# best-performance summaries AND the full-grid Supp. Fig. 1A/B
# (hap_throughput/track_throughput). FASTA/BigWig baselines are the
# apples-to-apples re-measurements on the SAME hardware (cn-03) as the 0.27 grid,
# in results_gvl027/baselines/ — NOT the old 0.6.1-era results/ CSVs (different
# hardware). The horizontal reference in every panel is cn-03's max sustained RAM
# bandwidth (STREAM Triad ~35 GB/s) — the real throughput ceiling on this
# hardware — replacing the old 31.5 GB/s A100 CPU->GPU PCIe line.

# %% hap data (GVL 0.27.0 eager grid + same-HW FASTA baseline)
results = gvl027_grid("results_gvl027/haps/*_none.csv")
ref_results = (
    gvl027_grid("results_gvl027/baselines/fasta.csv")
    .group_by("seqlen")
    .agg(pl.col("throughput").max())
)
results = results.join(
    ref_results.select("seqlen", ref_throughput="throughput"), on=["seqlen"]
).with_columns(
    pl.col("dataset").replace(GVL027_LABELS),
    batch_mb=pl.col("batch_size") * pl.col("seqlen") / 1e6,
)

# %% track data (GVL 0.27.0 eager grid + same-HW pyBigWig baseline)
track_results = gvl027_grid("results_gvl027/tracks/*_none.csv").with_columns(
    batch_mb=pl.col("batch_size") * pl.col("seqlen") * 4 / 1e6,
)
pybigwig_results = (
    gvl027_grid("results_gvl027/baselines/pybigwig.csv")
    .group_by("seqlen")
    .agg(pl.col("throughput").max())
)
track_results = track_results.join(
    pybigwig_results.select("seqlen", bigwig_throughput="throughput"), on="seqlen"
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
    RAM_BW_GBPS,
    c="k",
    ls="--",
    alpha=0.5,
    linewidth=3,
)
ax.text(
    results["n_nucleotides"].min() - 1500,  # pyright: ignore
    RAM_BW_GBPS,
    RAM_BW_LABEL,
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
    RAM_BW_GBPS,
    c="k",
    ls="--",
    alpha=0.5,
    linewidth=3,
)
ax.text(
    track_results["n_nucleotides"].min() - 3000,  # pyright: ignore
    RAM_BW_GBPS,
    RAM_BW_LABEL,
    va="center",
    ha="right",
)
fg.tight_layout()
fg.savefig(fig_dir / "track_throughput.svg")
fg.savefig(fig_dir / "track_throughput.png", dpi=150)

# %%
# best track results (GVL 0.27.0 eager vs same-HW pyBigWig baseline; RAM-bw ceiling)
gvl027_bigwig = gvl027_peak("results_gvl027/baselines/pybigwig.csv").sort("seqlen")
fig, ax = plt.subplots()
sns.lineplot(
    data=gvl027_peak("results_gvl027/tracks/*_none.csv").sort("seqlen").to_pandas(),
    x="seqlen",
    y="throughput",
    ax=ax,
    label="GVL",
    linewidth=5,
    solid_joinstyle="round",
    solid_capstyle="round",
)
sns.lineplot(
    data=gvl027_bigwig.to_pandas(),
    x="seqlen",
    y="throughput",
    ax=ax,
    color="C2",
    label="BigWig",
    linewidth=5,
    solid_joinstyle="round",
    solid_capstyle="round",
)
ax.axhline(RAM_BW_GBPS, c="k", ls="--", alpha=0.5, linewidth=5)
ax.text(
    gvl027_bigwig["seqlen"].min() - 1000,  # pyright: ignore
    RAM_BW_GBPS,
    RAM_BW_LABEL,
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

# %% best haplotype performance (GVL 0.27.0 eager vs same-HW FASTA baseline; RAM-bw ceiling)
gvl_haps = gvl027_peak("results_gvl027/haps/*_none.csv").with_columns(
    pl.col("dataset").replace(GVL027_LABELS)
)
gvl027_fasta = gvl027_peak("results_gvl027/baselines/fasta.csv")  # dataset == "FASTA"
data = pl.concat(
    [gvl_haps, gvl027_fasta],
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
ax.axhline(RAM_BW_GBPS, c="k", ls="--", alpha=0.5, linewidth=3)
ax.text(
    gvl027_fasta["seqlen"].min() - 1000,  # pyright: ignore
    RAM_BW_GBPS,
    RAM_BW_LABEL,
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
memory = disk_usage_df()
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
    pl.read_csv(proj_dir / "results" / "variants_batched_throughput_1kgp_par4_nb_gather.csv")
    .rename({"presubset_bcf_time": "presub-bcf_time"})
    .unpivot(
        ["svar_time", "bcf_time", "plink_time", "presub-bcf_time"],
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

q_len_name = r"$\log_{10}$ query length"
data = var_throughput.with_columns(
    pl.col("query_length").log(10),
    log_vars_per_sec=(pl.col("n_variants") / pl.col("time") * 1e9).log(10),
).rename({"query_length": q_len_name, "filetype": "File type"})
fg = sns.lmplot(
    data,
    x=q_len_name,
    y="log_vars_per_sec",
    hue="File type",
    aspect=0.6,
    lowess=True,
)
_ = fg.set(xlabel=q_len_name, ylabel=r"$\log_{10}$ variants/s")
sns.move_legend(fg, "center left", bbox_to_anchor=(0.95, 0.5))
fg.figure.tight_layout()
# fg.savefig(
#     fig_dir / "variant_throughput.png",
#     dpi=300,
#     bbox_inches="tight",
# )
# fg.savefig(fig_dir / "variant_throughput.svg")

# %%
var_throughput.head()
