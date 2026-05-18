#! /usr/bin/env python

from pathlib import Path

from cyclopts import run

from _plot_common import METHOD_LABELS, lmplot


def bench(*csvs: Path, output_dir: Path = Path("results")):
    import numpy as np
    import polars as pl
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.5)

    if not csvs:
        csvs = tuple(output_dir / f"{m}_throughput.csv" for m in METHOD_LABELS)

    schema_overrides = {"setup_ns": pl.Int64}
    raw = pl.concat([pl.read_csv(p, schema_overrides=schema_overrides) for p in csvs])

    # ---- q-len axis (rows where n_samples is at its max, i.e. full cohort) ----
    full_n = raw["n_samples"].max()
    qlen_df = (
        raw.filter(pl.col("n_samples") == full_n)
        .filter((pl.col("n_calls") > 0) & (pl.col("elapsed_ns") > 0))
        .with_columns(
            (pl.col("n_calls") / (pl.col("elapsed_ns") * 1e-9)).alias("calls_per_sec"),
        )
        .with_columns(
            pl.col("query_length").log(base=10).alias("log10_query_length"),
            pl.col("calls_per_sec").log(base=10).alias("log10_calls_per_sec"),
            pl.col("method").replace(METHOD_LABELS).alias("method_label"),
        )
        .to_pandas()
    )

    if not qlen_df.empty:
        hue_order = (
            qlen_df.groupby("method_label")["calls_per_sec"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        lmplot(
            qlen_df,
            x_col="log10_query_length",
            y_col="log10_calls_per_sec",
            x_label="Query length (bp)",
            y_label="Throughput (alt calls / sec)",
            hue_order=hue_order,
            output_dir=output_dir,
            stem="plot",
        )

    # ---- N axis (rows where query_length is at its max in the N sweep — i.e. the
    #      fixed N-sweep query length; we identify it as the query_length value
    #      whose rows contain more than one distinct n_samples value) ----
    n_distinct_by_q = raw.group_by("query_length").agg(
        pl.col("n_samples").n_unique().alias("n_unique_samples")
    )
    n_sweep_qlens = n_distinct_by_q.filter(pl.col("n_unique_samples") > 1)["query_length"].to_list()

    if n_sweep_qlens:
        n_df = (
            raw.filter(pl.col("query_length").is_in(n_sweep_qlens))
            .filter((pl.col("n_calls") > 0) & (pl.col("elapsed_ns") > 0))
            .with_columns(
                (pl.col("n_calls") / (pl.col("elapsed_ns") * 1e-9)).alias("calls_per_sec"),
            )
            .with_columns(
                pl.col("n_samples").log(base=10).alias("log10_n_samples"),
                pl.col("calls_per_sec").log(base=10).alias("log10_calls_per_sec"),
                pl.col("method").replace(METHOD_LABELS).alias("method_label"),
            )
            .to_pandas()
        )
        if not n_df.empty:
            hue_order_n = (
                n_df.groupby("method_label")["calls_per_sec"]
                .max()
                .sort_values(ascending=False)
                .index.tolist()
            )
            lmplot(
                n_df,
                x_col="log10_n_samples",
                y_col="log10_calls_per_sec",
                x_label="Cohort size (N samples)",
                y_label="Throughput (alt calls / sec)",
                hue_order=hue_order_n,
                output_dir=output_dir,
                stem="n_plot",
            )

    # ---- Setup-cost plot, q-len axis only (existing behavior, full cohort) ----
    setup_df = (
        raw.filter(pl.col("n_samples") == full_n)
        .filter(
            (pl.col("n_calls") > 0)
            & pl.col("setup_ns").is_not_null()
            & (pl.col("setup_ns") > 0)
        )
        .with_columns(
            (pl.col("n_calls") / (pl.col("setup_ns") * 1e-9)).alias("setup_calls_per_sec"),
        )
        .with_columns(
            pl.col("query_length").log(base=10).alias("log10_query_length"),
            pl.col("setup_calls_per_sec").log(base=10).alias("log10_setup_calls_per_sec"),
            pl.col("method").replace(METHOD_LABELS).alias("method_label"),
        )
        .to_pandas()
    )

    if not setup_df.empty:
        setup_hue_order = (
            setup_df.groupby("method_label")["setup_calls_per_sec"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        lmplot(
            setup_df,
            x_col="log10_query_length",
            y_col="log10_setup_calls_per_sec",
            x_label="Query length (bp)",
            y_label="Setup throughput (alt calls / sec)",
            hue_order=setup_hue_order,
            output_dir=output_dir,
            stem="setup_plot",
        )


if __name__ == "__main__":
    run(bench)
