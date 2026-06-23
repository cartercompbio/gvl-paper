#! /usr/bin/env python
"""Combine the four variant random-access benchmark plots into one 2x2 panel.

Reads ``variant_throughput/results/{method}_{throughput,memory}.csv`` and draws:

    (a) memory_plot    peak RSS vs query length   (full cohort)
    (b) setup_plot     setup throughput vs qlen   (SVAR, PRESUB-BCF only)
    (c) n_memory_plot  peak RSS vs cohort size    (fixed query length)
    (d) n_plot         read throughput vs cohort  (fixed query length)

Method colors are fixed across all panels and a single legend is shared by the
whole figure, so SVAR/BCF/PGEN/PRESUB-BCF are the same color everywhere.
"""

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "variant_throughput" / "bin"))

from cyclopts import run

from _plot_common import METHOD_LABELS  # noqa: E402

# Fixed method order and palette -> consistent colors in every subpanel.
METHOD_ORDER = ["SVAR", "BCF", "PGEN", "PRESUB-BCF"]


def _palette():
    import seaborn as sns

    base = sns.color_palette("colorblind", n_colors=len(METHOD_ORDER))
    return dict(zip(METHOD_ORDER, base))


def _log_panel(ax, df, x_col, y_col, x_label, y_label, palette, title):
    """LOWESS fit per method onto a shared (already log10) Axes."""
    import numpy as np
    import seaborn as sns

    present = [m for m in METHOD_ORDER if m in set(df["method_label"])]
    for method in present:
        sub = df[df["method_label"] == method]
        sns.regplot(
            data=sub,
            x=x_col,
            y=y_col,
            lowess=True,
            ax=ax,
            color=palette[method],
            scatter_kws={"s": 28, "alpha": 0.5},
            line_kws={"linewidth": 3},
            label=method,
        )

    xticks = np.arange(np.floor(df[x_col].min()), np.ceil(df[x_col].max()) + 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"$10^{{{int(x)}}}$" for x in xticks])
    yticks = np.arange(np.floor(df[y_col].min()), np.ceil(df[y_col].max()) + 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$10^{{{int(y)}}}$" for y in yticks])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontweight="bold")


def bench(results_dir: Path = Path("variant_throughput/results"), output_dir: Path = Path("figures")):
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    palette = _palette()

    methods = list(METHOD_LABELS)
    tp = pl.concat(
        [pl.read_csv(results_dir / f"{m}_throughput.csv", schema_overrides={"setup_ns": pl.Int64}) for m in methods]
    )
    mem = pl.concat([pl.read_csv(results_dir / f"{m}_memory.csv") for m in methods])

    full_n = tp["n_samples"].max()

    def label(col):
        return pl.col("method").replace(METHOD_LABELS).alias("method_label")

    # (a) peak RSS vs query length, full cohort
    mem_q = (
        mem.filter((pl.col("n_samples") == full_n) & (pl.col("n_calls") > 0) & (pl.col("peak_rss_bytes") > 0))
        .with_columns(
            pl.col("query_length").log(base=10).alias("x"),
            (pl.col("peak_rss_bytes") / 2**20).log(base=10).alias("y"),
            label("method"),
        )
        .to_pandas()
    )

    # (b) setup throughput vs query length, full cohort (only methods with a setup phase)
    setup_q = (
        tp.filter(
            (pl.col("n_samples") == full_n)
            & (pl.col("n_calls") > 0)
            & pl.col("setup_ns").is_not_null()
            & (pl.col("setup_ns") > 0)
        )
        .with_columns(
            pl.col("query_length").log(base=10).alias("x"),
            (pl.col("n_calls") / (pl.col("setup_ns") * 1e-9)).log(base=10).alias("y"),
            label("method"),
        )
        .to_pandas()
    )

    # cohort-size sweep: the query length that carries more than one distinct n_samples
    n_distinct = mem.group_by("query_length").agg(pl.col("n_samples").n_unique().alias("u"))
    n_qlens = n_distinct.filter(pl.col("u") > 1)["query_length"].to_list()

    # (c) peak RSS vs cohort size
    mem_n = (
        mem.filter(pl.col("query_length").is_in(n_qlens) & (pl.col("n_calls") > 0) & (pl.col("peak_rss_bytes") > 0))
        .with_columns(
            pl.col("n_samples").log(base=10).alias("x"),
            (pl.col("peak_rss_bytes") / 2**20).log(base=10).alias("y"),
            label("method"),
        )
        .to_pandas()
    )

    # (d) read throughput vs cohort size
    tp_n = (
        tp.filter(pl.col("query_length").is_in(n_qlens) & (pl.col("n_calls") > 0) & (pl.col("elapsed_ns") > 0))
        .with_columns(
            pl.col("n_samples").log(base=10).alias("x"),
            (pl.col("n_calls") / (pl.col("elapsed_ns") * 1e-9)).log(base=10).alias("y"),
            label("method"),
        )
        .to_pandas()
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    _log_panel(axes[0, 0], mem_q, "x", "y", "Query length (bp)", "Peak RSS (MiB)", palette, "a")
    _log_panel(axes[0, 1], setup_q, "x", "y", "Query length (bp)", "Setup throughput (alt calls / sec)", palette, "b")
    _log_panel(axes[1, 0], mem_n, "x", "y", "Cohort size (N samples)", "Peak RSS (MiB)", palette, "c")
    _log_panel(axes[1, 1], tp_n, "x", "y", "Cohort size (N samples)", "Throughput (alt calls / sec)", palette, "d")

    # one consolidated legend for the whole figure
    handles = [plt.Line2D([], [], color=palette[m], marker="o", linestyle="-", label=m) for m in METHOD_ORDER]
    fig.legend(
        handles=handles,
        title="Method",
        loc="lower center",
        ncol=len(METHOD_ORDER),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "variant_throughput_panel"
    for fmt in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_dir / stem}.{{png,svg,pdf}}")


if __name__ == "__main__":
    run(bench)
