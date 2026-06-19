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

import matplotlib.legend as mlegend
import numpy as np
from matplotlib.patches import Patch, Rectangle
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
    variant_qlen_df,
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


def _ram_bw_line(ax, xmax):
    # Label at the right end of the line: the top-right corner is empty (curves
    # peak ~19 GB/s, below the 35 GB/s ceiling) and clear of the a)-f) letters.
    ax.axhline(RAM_BW_GBPS, c="k", ls="--", alpha=0.5, linewidth=2)
    ax.text(xmax, RAM_BW_GBPS, RAM_BW_LABEL, va="bottom", ha="right", fontsize=7)


def _clean_legend(ax, keep_labels, **kw):
    """Build one legend from explicit handles, keeping only `keep_labels` in order.

    Avoids ultraplot's auto-collecting legend, which otherwise scrapes seaborn's
    internal artists (e.g. 'y', 'ymin', 'width') and duplicates the legend box.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, lab in zip(handles, labels):
        by_label.setdefault(lab, h)
    # Hide every pre-existing legend before adding ours. ultraplot auto-creates a
    # legend from labeled artists and stores it as an axes child (NOT ax.legend_),
    # polluted with seaborn internals ('y', 'ymin', 'width'). Legends can't be
    # .remove()'d on ultraplot axes, so hide them.
    if getattr(ax, "legend_", None) is not None:
        ax.legend_.set_visible(False)
    for child in list(ax.get_children()):
        if isinstance(child, mlegend.Legend):
            child.set_visible(False)
    ordered = [(by_label[lab], lab) for lab in keep_labels if lab in by_label]
    if ordered:
        hs, ls = zip(*ordered)
        ax.legend(list(hs), list(ls), **kw)


def panel_disk(ax):
    df = disk_usage_df()
    datasets = df["Dataset"].unique(maintain_order=True).to_list()
    impls = ["GVL", "FASTA"]
    colors = dict(zip(impls, sns.color_palette(n_colors=2)))
    floor = 0.05  # left edge of bars (log axis can't start at 0)
    th = 0.4  # bar thickness in y
    ax.set_xscale("log")
    # ultraplot monkeypatches bar/barh at the matplotlib class level with a
    # different arg convention, mangling both seaborn's call and matplotlib's
    # barh (bars come out vertical, with the data value as the y-extent). Draw
    # the grouped horizontal bars as plain Rectangle patches, which no wrapper
    # intercepts: x from floor to value, y a fixed-thickness slab per dataset.
    for i, impl in enumerate(impls):
        for j, ds in enumerate(datasets):
            v = df.filter(
                (pl.col("Dataset") == ds) & (pl.col("Implementation") == impl)
            )["Disk Space (GB)"].item()
            ax.add_patch(
                Rectangle((floor, j + (i - 1) * th), v - floor, th, color=colors[impl])
            )
    short = {
        "TCGA BRCA ATAC (n=62)": "TCGA ATAC (62)",
        "1000 Genomes (n=3,202)": "1000G (3,202)",
        "Biobank, chr22 (n=487,409)": "Biobank (487k)",
    }
    # Explicit xlim/ylim: ultraplot's autoscale ignores add_patch extents (it
    # otherwise blew the y-range up to +/-3851). Reversed ylim = first dataset on top.
    ax.format(
        xlabel="Disk space (GB)",
        ylabel="",
        title="Storage",
        xlim=(floor, df["Disk Space (GB)"].max() * 2),
        yticks=list(np.arange(len(datasets))),
        yticklabels=[short[d] for d in datasets],
        ylim=(len(datasets) - 0.5, -0.5),
    )
    for child in list(ax.get_children()):
        if isinstance(child, mlegend.Legend):
            child.set_visible(False)
    ax.legend([Patch(color=colors[k], label=k) for k in impls], impls, loc="lr", ncols=1)


def panel_variant(ax):
    df = variant_qlen_df().to_pandas()
    palette = dict(zip(VARIANT_ORDER, sns.color_palette(n_colors=len(VARIANT_ORDER))))
    for method in VARIANT_ORDER:
        sub = df[df["method_label"] == method]
        if sub.empty:
            continue
        sns.regplot(
            data=sub,
            x="log10_query_length",
            y="log10_calls_per_sec",
            lowess=True,
            ax=ax,
            label=method,
            color=palette[method],
            scatter_kws=dict(s=12, alpha=0.4),
            line_kws=dict(linewidth=2),
        )
    ax.format(
        xlabel=r"$\log_{10}$ query length (bp)",
        ylabel=r"$\log_{10}$ variant calls/s",
        title="Variant query throughput",
    )
    _clean_legend(ax, VARIANT_ORDER, loc="ur", ncols=1, fontsize=7)


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
    _ram_bw_line(ax, gvl["seqlen"].max())
    ax.format(
        xscale="log", yscale="log", xlabel="Sequence length",
        ylabel="Throughput (GB/s)", title="Haplotypes vs FASTA",
    )
    _clean_legend(ax, HAP_HUE_ORDER, loc="lr", ncols=1, fontsize=6)


def panel_tracks(ax):
    gvl = gvl027_peak("results_gvl027/tracks/*_none.csv").sort("seqlen")
    bw = gvl027_peak("results_gvl027/baselines/pybigwig.csv").sort("seqlen")
    sns.lineplot(gvl.to_pandas(), x="seqlen", y="throughput", ax=ax, label="GVL",
                 linewidth=2.5, solid_joinstyle="round", solid_capstyle="round")
    sns.lineplot(bw.to_pandas(), x="seqlen", y="throughput", ax=ax, label="BigWig",
                 color="C2", linewidth=2.5, solid_joinstyle="round", solid_capstyle="round")
    _ram_bw_line(ax, bw["seqlen"].max())
    ax.format(
        xscale="log", yscale="log", xlabel="Sequence length",
        ylabel="Throughput (GB/s)", title="Tracks vs BigWig",
    )
    _clean_legend(ax, ["GVL", "BigWig"], loc="lr", ncols=1)


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
    _clean_legend(
        ax,
        [r"$\rho$ across genes", r"$\rho$ across individuals"],
        loc="ul",
        ncols=1,
        fontsize=6,
    )


def main():
    fig, axs = uplt.subplots(nrows=2, ncols=3, refwidth=2.3, share=False)
    fig.format(abc="a)", abcloc="ul")
    panel_disk(axs[0])
    panel_variant(axs[1])
    panel_haps(axs[2])
    panel_tracks(axs[3])
    panel_gpu_placeholder(axs[4])
    panel_basenji2(axs[5])
    fig.save(fig_dir / "figure2.svg")
    fig.save(fig_dir / "figure2.png", dpi=200)
    print("wrote", fig_dir / "figure2.svg")


if __name__ == "__main__":
    main()
