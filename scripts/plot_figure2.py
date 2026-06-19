#! /usr/bin/env python3
"""Assemble manuscript Figure 2 data panels into one vector figures/figure2.svg.

Panels: a) disk usage, b) variant calls/sec vs cohort N, c) haplotype
throughput, d) track throughput, e) reserved GPU-utilization slot (composited
separately), f) Basenji2 rho ECDF. ultraplot owns layout + a)-f) lettering;
seaborn axes-level functions draw each panel. See
docs/superpowers/specs/2026-06-18-figure2-unified-svg-design.md.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import ultraplot as uplt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_data import (  # noqa: E402
    GVL027_LABELS,
    RAM_BW_GBPS,
    RAM_BW_LABEL,
    disk_usage_df,
    gvl027_peak,
    variant_n_df,
)

proj_dir = Path(__file__).resolve().parent.parent
fig_dir = proj_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

HAP_HUE_ORDER = [
    "GVL: TCGA BRCA ATAC (n=62)",
    "GVL: 1000 Genomes (n=3,202)",
    "GVL: Biobank (n=487,409)",
    "FASTA",
]
VARIANT_ORDER = ["SVAR", "PRESUB-BCF", "BCF", "PGEN"]


def _ram_bw_line(ax, xmin):
    ax.axhline(RAM_BW_GBPS, c="k", ls="--", alpha=0.5, linewidth=2)
    ax.text(xmin, RAM_BW_GBPS, RAM_BW_LABEL, va="bottom", ha="left", fontsize=7)


def panel_disk(ax):
    df = disk_usage_df().to_pandas()
    sns.barplot(df, x="Disk Space (GB)", y="Dataset", hue="Implementation", ax=ax)
    ax.format(xscale="log", xlabel="Disk space (GB)", ylabel="", title="Storage")
    ax.legend(loc="lr", ncols=1)


def panel_variant_n(ax):
    df = variant_n_df().to_pandas()
    palette = dict(zip(VARIANT_ORDER, sns.color_palette(n_colors=len(VARIANT_ORDER))))
    for method in VARIANT_ORDER:
        sub = df[df["method_label"] == method]
        if sub.empty:
            continue
        sns.regplot(
            data=sub,
            x="log10_n_samples",
            y="log10_calls_per_sec",
            lowess=True,
            ax=ax,
            label=method,
            color=palette[method],
            scatter_kws=dict(s=12, alpha=0.4),
            line_kws=dict(linewidth=2),
        )
    ax.format(
        xlabel=r"$\log_{10}$ cohort size (N)",
        ylabel=r"$\log_{10}$ variant calls/s",
        title="Variant query throughput",
    )
    ax.legend(loc="lr", ncols=1)


def panel_haps(ax):
    gvl = gvl027_peak("results_gvl027/haps/*_none.csv").with_columns(
        pl.col("dataset").replace(GVL027_LABELS)
    )
    fasta = gvl027_peak("results_gvl027/baselines/fasta.csv")
    data = pl.concat([gvl, fasta], how="diagonal_relaxed").rename({"dataset": "Dataset"})
    sns.lineplot(
        data.to_pandas(),
        x="seqlen",
        y="throughput",
        hue="Dataset",
        hue_order=HAP_HUE_ORDER,
        ax=ax,
        linewidth=2.5,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    _ram_bw_line(ax, gvl["seqlen"].min())
    ax.format(
        xscale="log", yscale="log", xlabel="Sequence length",
        ylabel="Throughput (GB/s)", title="Haplotypes vs FASTA",
    )
    ax.legend(loc="lr", ncols=1, fontsize=6)


def panel_tracks(ax):
    gvl = gvl027_peak("results_gvl027/tracks/*_none.csv").sort("seqlen")
    bw = gvl027_peak("results_gvl027/baselines/pybigwig.csv").sort("seqlen")
    sns.lineplot(gvl.to_pandas(), x="seqlen", y="throughput", ax=ax, label="GVL",
                 linewidth=2.5, solid_joinstyle="round", solid_capstyle="round")
    sns.lineplot(bw.to_pandas(), x="seqlen", y="throughput", ax=ax, label="BigWig",
                 color="C2", linewidth=2.5, solid_joinstyle="round", solid_capstyle="round")
    _ram_bw_line(ax, bw["seqlen"].min())
    ax.format(
        xscale="log", yscale="log", xlabel="Sequence length",
        ylabel="Throughput (GB/s)", title="Tracks vs BigWig",
    )
    ax.legend(loc="lr", ncols=1)


def panel_gpu_placeholder(ax):
    ax.format(title="GPU utilization", xlabel="", ylabel="")
    ax.format(xticks=[], yticks=[])
    ax.text(
        0.5, 0.5, "GPU utilization\n(composited separately)",
        ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=8,
    )


def panel_basenji2(ax):
    cache = fig_dir / "basenji2_rho.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} missing — run scripts/plot_basenji2.py first "
            "(see run_scripts.sh) to generate the rho cache."
        )
    d = np.load(cache)
    gene_rho, indiv_rho = d["gene_rho"].ravel(), d["indiv_rho"].ravel()
    sns.ecdfplot(gene_rho, label=r"$\rho$ across genes", ax=ax, linewidth=2.5)
    ax.axvline(np.nanmean(gene_rho), c="k", ls="--", alpha=0.5, linewidth=2)
    sns.ecdfplot(indiv_rho, label=r"$\rho$ across individuals", ax=ax, linewidth=2.5)
    ax.axvline(np.nanmean(indiv_rho), c="k", ls="--", alpha=0.5, linewidth=2)
    ax.format(xlabel=r"Spearman $\rho$", ylabel="Proportion", title="Basenji2 evaluation")
    ax.legend(loc="ul", ncols=1, fontsize=6)


def main():
    fig, axs = uplt.subplots(nrows=2, ncols=3, refwidth=2.3, share=False)
    fig.format(abc="a)", abcloc="ul")
    panel_disk(axs[0])
    panel_variant_n(axs[1])
    panel_haps(axs[2])
    panel_tracks(axs[3])
    panel_gpu_placeholder(axs[4])
    panel_basenji2(axs[5])
    fig.save(fig_dir / "figure2.svg")
    fig.save(fig_dir / "figure2.png", dpi=200)
    print("wrote", fig_dir / "figure2.svg")


if __name__ == "__main__":
    main()
