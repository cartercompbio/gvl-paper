#! /usr/bin/env python
"""Join the 0.26.0 probe results to the v0.6.1 baseline and report parity.

Outputs:
  results_gvl026/parity_summary.csv  — per (dataset,output_mode,dl_mode,threads,seqlen,batch_size):
       baseline MiB/s, 0.26.0 MiB/s (median over replicates), ratio = v026 / v061
  figures/gvl026_parity.png          — scatter, x=baseline, y=0.26.0, hue=dl_mode, facet=seqlen
"""

from pathlib import Path

import cyclopts


def main(
    results_dir: Path = Path("results_gvl026"),
    hap_baseline: Path = Path("results/hap_results.csv"),
    track_baseline: Path = Path("results/track_results.csv"),
    fig_out: Path = Path("figures/gvl026_parity.png"),
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    # --- load probe results ---
    probe = pl.concat(
        [pl.read_csv(p) for p in sorted(results_dir.glob("*_*_*_*.csv"))],
        how="vertical_relaxed",
    ).rename({"throughput (MiB/s)": "v026"})
    # benchmark_dl.py writes the literal "nan" for skipped cells, which makes
    # polars infer the column as String. Coerce to Float and treat NaN as null.
    probe = (
        probe.with_columns(
            pl.col("v026").cast(pl.Float64, strict=False).fill_nan(None)
        )
        .drop_nulls("v026")
        .group_by(["dataset", "mode", "dl_mode", "threads", "seqlen", "batch_size"])
        .agg(pl.col("v026").median())
    )

    # --- load baselines, tag output mode ---
    hap = pl.read_csv(hap_baseline).with_columns(mode=pl.lit("haps"))
    trk = pl.read_csv(track_baseline).with_columns(mode=pl.lit("tracks"))
    base = (
        pl.concat([hap, trk], how="vertical_relaxed")
        .rename({"throughput (MiB/s)": "v061"})
        .group_by(["dataset", "mode", "threads", "seqlen", "batch_size"])
        .agg(pl.col("v061").median())
    )

    joined = probe.join(
        base, on=["dataset", "mode", "threads", "seqlen", "batch_size"], how="left"
    ).with_columns(ratio=(pl.col("v026") / pl.col("v061")))

    results_dir.mkdir(exist_ok=True)
    summary = results_dir / "parity_summary.csv"
    joined.sort(["dataset", "mode", "dl_mode", "seqlen", "batch_size", "threads"]).write_csv(summary)
    print(f"WROTE {summary} ({joined.height} rows)")

    matched = joined.drop_nulls("v061")
    print("Median 0.26.0/0.6.1 ratio by dl_mode:")
    print(matched.group_by("dl_mode").agg(pl.col("ratio").median()).sort("dl_mode"))

    # --- parity scatter ---
    pdf = matched.to_pandas()
    fig_out.parent.mkdir(exist_ok=True)
    g = sns.relplot(
        data=pdf, x="v061", y="v026", hue="dl_mode", style="mode",
        col="seqlen", col_wrap=2, facet_kws={"sharex": False, "sharey": False},
    )
    for ax in g.axes.flat:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], ls="--", c="grey", lw=1)  # parity line
        ax.set_xscale("log"); ax.set_yscale("log")
    g.set_axis_labels("v0.6.1 MiB/s", "v0.26.0 MiB/s")
    g.savefig(fig_out, dpi=150, bbox_inches="tight")
    print(f"WROTE {fig_out}")


if __name__ == "__main__":
    cyclopts.run(main)
