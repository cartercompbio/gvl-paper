#! /usr/bin/env python
"""Plot peak and average RAM (RSS) during SVAR conversion, per dataset.

Reads the per-dataset conversion-memory CSVs emitted by the gvl027 full bench
(``results_gvl027/svar_convert_memory/{dataset}.csv``, columns
``dataset,source_fmt,n_jobs,avg_rss_bytes,peak_rss_bytes``) and renders a
grouped bar chart of average vs. peak RSS in GiB. This is the SVAR-conversion
counterpart to the dataloading memory-growth figure (Supplementary Figure 2);
per reviewer R2.1 we report peak and average here rather than a growth curve.
"""

from pathlib import Path

from cyclopts import run


def bench(
    *csvs: Path,
    input_dir: Path = Path("results_gvl027/svar_convert_memory"),
    output_dir: Path = Path("results_gvl027"),
    stem: str = "convert_memory_plot",
    alloc_gib: float = 128.0,
):
    """alloc_gib: per-job memory allocated to BENCH_SVAR_CONVERT (Nextflow
    ``memory 128.GB``), drawn as a dashed reference line."""
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.5)

    if not csvs:
        csvs = tuple(sorted(input_dir.glob("*.csv")))
    if not csvs:
        raise SystemExit(f"no conversion-memory CSVs found in {input_dir}")

    raw = pl.concat([pl.read_csv(p) for p in csvs])

    # bytes -> GiB, then long form: one row per (dataset, statistic)
    long = (
        raw.with_columns(
            (pl.col("avg_rss_bytes") / 2**30).alias("Average"),
            (pl.col("peak_rss_bytes") / 2**30).alias("Peak"),
        )
        .unpivot(
            on=["Average", "Peak"],
            index=["dataset", "source_fmt"],
            variable_name="Statistic",
            value_name="rss_gib",
        )
        # label the source format alongside the dataset name
        .with_columns(
            (pl.col("dataset") + " (" + pl.col("source_fmt") + ")").alias("dataset_label")
        )
        .sort("dataset_label", "Statistic")
        .to_pandas()
    )

    dataset_order = sorted(long["dataset_label"].unique())

    fig, ax = plt.subplots(figsize=(max(4, 1.8 * len(dataset_order) + 2), 5))
    sns.barplot(
        data=long,
        x="dataset_label",
        y="rss_gib",
        hue="Statistic",
        hue_order=["Average", "Peak"],
        order=dataset_order,
        ax=ax,
    )
    ax.axhline(
        alloc_gib,
        ls="--",
        color="red",
        lw=1.5,
        label=f"Allocated ({alloc_gib:g} GiB)",
    )
    ax.set_xlabel("Dataset (source format)")
    ax.set_ylabel("RSS during SVAR conversion (GiB)")
    ax.legend(title="")

    # annotate bars with their GiB value
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=11)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_dir}/{stem}.{{png,svg,pdf}}")


if __name__ == "__main__":
    run(bench)
