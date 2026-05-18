#! /usr/bin/env python

from pathlib import Path

from cyclopts import run

from _plot_common import METHOD_LABELS, lmplot


def bench(*csvs: Path, output_dir: Path = Path("results")):
    import polars as pl
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.5)

    if not csvs:
        csvs = tuple(output_dir / f"{m}_memory.csv" for m in METHOD_LABELS)

    raw = pl.concat([pl.read_csv(p) for p in csvs])

    # ---- q-len axis: full-cohort rows only ----
    full_n = raw["n_samples"].max()
    qlen_df = (
        raw.filter(pl.col("n_samples") == full_n)
        .filter((pl.col("n_calls") > 0) & (pl.col("peak_rss_bytes") > 0))
        .with_columns((pl.col("peak_rss_bytes") / 2**20).alias("peak_rss_mib"))
        .with_columns(
            pl.col("query_length").log(base=10).alias("log10_query_length"),
            pl.col("peak_rss_mib").log(base=10).alias("log10_peak_rss_mib"),
            pl.col("method").replace(METHOD_LABELS).alias("method_label"),
        )
        .to_pandas()
    )

    if not qlen_df.empty:
        hue_order = (
            qlen_df.groupby("method_label")["peak_rss_mib"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        lmplot(
            qlen_df,
            x_col="log10_query_length",
            y_col="log10_peak_rss_mib",
            x_label="Query length (bp)",
            y_label="Peak RSS (MiB)",
            hue_order=hue_order,
            output_dir=output_dir,
            stem="memory_plot",
        )

    # ---- N axis: rows whose query_length has more than one distinct n_samples ----
    n_distinct_by_q = raw.group_by("query_length").agg(
        pl.col("n_samples").n_unique().alias("n_unique_samples")
    )
    n_sweep_qlens = n_distinct_by_q.filter(pl.col("n_unique_samples") > 1)["query_length"].to_list()

    if n_sweep_qlens:
        n_df = (
            raw.filter(pl.col("query_length").is_in(n_sweep_qlens))
            .filter((pl.col("n_calls") > 0) & (pl.col("peak_rss_bytes") > 0))
            .with_columns((pl.col("peak_rss_bytes") / 2**20).alias("peak_rss_mib"))
            .with_columns(
                pl.col("n_samples").log(base=10).alias("log10_n_samples"),
                pl.col("peak_rss_mib").log(base=10).alias("log10_peak_rss_mib"),
                pl.col("method").replace(METHOD_LABELS).alias("method_label"),
            )
            .to_pandas()
        )
        if not n_df.empty:
            hue_order_n = (
                n_df.groupby("method_label")["peak_rss_mib"]
                .max()
                .sort_values(ascending=False)
                .index.tolist()
            )
            lmplot(
                n_df,
                x_col="log10_n_samples",
                y_col="log10_peak_rss_mib",
                x_label="Cohort size (N samples)",
                y_label="Peak RSS (MiB)",
                hue_order=hue_order_n,
                output_dir=output_dir,
                stem="n_memory_plot",
            )


if __name__ == "__main__":
    run(bench)
