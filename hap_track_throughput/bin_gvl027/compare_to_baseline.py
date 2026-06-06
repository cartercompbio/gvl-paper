#! /usr/bin/env python
"""Compare the GVL 0.27.0 full-bench results to the v0.6.1 baseline.

Reads the Nextflow output layout under results_gvl027/:
    haps/{dataset}_{length}_{backend}_{dl_mode}.csv          (throughput)
    tracks/{dataset}_{length}_{backend}_{dl_mode}.csv        (throughput)
    haps_memory/...   tracks_memory/...                      (peak/avg RSS)

Outputs:
  results_gvl027/parity_summary.csv  — throughput, dl_mode=="none" joined to the
       0.6.1 baseline on (dataset,mode,threads,seqlen,batch_size); ratio = v027/v061.
       INTERNAL sanity check, not a paper deliverable.
  results_gvl027/buffered_throughput.csv — buffered throughput, standalone (no baseline).
  results_gvl027/memory_summary.csv  — per (dataset,mode,dl_mode,seqlen,batch_size)
       peak/avg RSS, absolute (no baseline join).
  figures/gvl027_parity.png          — v061-vs-v027 throughput scatter (none mode).
  figures/gvl027_peak_rss.png        — peak RSS vs batch_size, faceted by seqlen.
"""

from pathlib import Path

import cyclopts


def norm_dataset(name: str) -> str:
    """Normalize a dataset label to the baseline convention (lower, '_'->'-')."""
    return name.lower().replace("_", "-")


def parse_mode_dir(dirname: str) -> tuple[str, bool]:
    """Map an nf output subdir to (output_mode, is_memory).

    'haps'->('haps',False); 'haps_memory'->('haps',True); same for 'tracks'.
    Raises ValueError for anything else (e.g. 'svar_convert', 'write').
    """
    if dirname in ("haps", "tracks"):
        return dirname, False
    if dirname in ("haps_memory", "tracks_memory"):
        return dirname[: -len("_memory")], True
    raise ValueError(f"not a haps/tracks result dir: {dirname!r}")


def _load_dir(results_dir: Path, dirname: str):
    """Concat all CSVs in results_dir/dirname, tagging output mode + dataset norm."""
    import polars as pl

    mode, _is_mem = parse_mode_dir(dirname)
    d = results_dir / dirname
    files = sorted(d.glob("*.csv"))
    if not files:
        return None
    df = pl.concat([pl.read_csv(p) for p in files], how="vertical_relaxed")
    return df.with_columns(
        mode=pl.lit(mode),
        dataset_norm=pl.col("dataset").str.to_lowercase().str.replace_all("_", "-", literal=True),
    )


def main(
    results_dir: Path = Path("results_gvl027"),
    hap_baseline: Path = Path("results/hap_results.csv"),
    track_baseline: Path = Path("results/track_results.csv"),
    fig_dir: Path = Path("figures"),
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (imported for side effects / future use)
    import polars as pl
    import seaborn as sns

    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # ---------- throughput ----------
    tput_frames = [
        f for f in (_load_dir(results_dir, d) for d in ("haps", "tracks")) if f is not None
    ]
    if tput_frames:
        tput = pl.concat(tput_frames, how="vertical_relaxed").rename(
            {"throughput (MiB/s)": "v027"}
        )
        # NaN cells (buffered skip / empty epoch) come in as numeric NaN or "nan".
        tput = (
            tput.with_columns(pl.col("v027").cast(pl.Float64, strict=False).fill_nan(None))
            .drop_nulls("v027")
            .group_by(["dataset_norm", "mode", "dl_mode", "threads", "seqlen", "batch_size"])
            .agg(pl.col("v027").median())
        )

        # baselines
        hap = pl.read_csv(hap_baseline).with_columns(mode=pl.lit("haps"))
        trk = pl.read_csv(track_baseline).with_columns(mode=pl.lit("tracks"))
        base = (
            pl.concat([hap, trk], how="vertical_relaxed")
            .rename({"throughput (MiB/s)": "v061"})
            .with_columns(
                dataset_norm=pl.col("dataset").str.to_lowercase().str.replace_all("_", "-", literal=True)
            )
            .group_by(["dataset_norm", "mode", "threads", "seqlen", "batch_size"])
            .agg(pl.col("v061").median())
        )

        # internal parity: dl_mode == none vs baseline
        none = tput.filter(pl.col("dl_mode") == "none")
        joined = none.join(
            base, on=["dataset_norm", "mode", "threads", "seqlen", "batch_size"], how="left"
        ).with_columns(ratio=(pl.col("v027") / pl.col("v061")))
        summary = results_dir / "parity_summary.csv"
        joined.sort(["dataset_norm", "mode", "seqlen", "batch_size", "threads"]).write_csv(summary)
        print(f"WROTE {summary} ({joined.height} rows)")

        matched = joined.drop_nulls("v061")
        if matched.height:
            print("Median 0.27.0/0.6.1 throughput ratio by (dataset, mode):")
            print(
                matched.group_by(["dataset_norm", "mode"])
                .agg(pl.col("ratio").median())
                .sort(["dataset_norm", "mode"])
            )
            pdf = matched.to_pandas()
            g = sns.relplot(
                data=pdf, x="v061", y="v027", hue="dataset_norm", style="mode",
                col="seqlen", col_wrap=2, facet_kws={"sharex": False, "sharey": False},
            )
            for ax in g.axes.flat:
                lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
                hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot([lo, hi], [lo, hi], ls="--", c="grey", lw=1)
                ax.set_xscale("log"); ax.set_yscale("log")
            g.set_axis_labels("v0.6.1 MiB/s", "v0.27.0 MiB/s")
            g.savefig(fig_dir / "gvl027_parity.png", dpi=150, bbox_inches="tight")
            print(f"WROTE {fig_dir / 'gvl027_parity.png'}")

        # buffered standalone
        buffered = tput.filter(pl.col("dl_mode") == "buffered")
        buf_out = results_dir / "buffered_throughput.csv"
        buffered.sort(["dataset_norm", "mode", "seqlen", "batch_size", "threads"]).write_csv(buf_out)
        print(f"WROTE {buf_out} ({buffered.height} rows)")
    else:
        print("No throughput CSVs found under haps/ or tracks/ — skipping throughput report.")

    # ---------- memory ----------
    mem_frames = [
        f
        for f in (_load_dir(results_dir, d) for d in ("haps_memory", "tracks_memory"))
        if f is not None
    ]
    if mem_frames:
        mem = pl.concat(mem_frames, how="vertical_relaxed")
        mem = (
            mem.with_columns(
                pl.col("peak_rss_bytes").cast(pl.Float64, strict=False).fill_nan(None),
                pl.col("avg_rss_bytes").cast(pl.Float64, strict=False).fill_nan(None),
            )
            .drop_nulls("peak_rss_bytes")
            .group_by(["dataset_norm", "mode", "dl_mode", "seqlen", "batch_size"])
            .agg(
                pl.col("peak_rss_bytes").max().alias("peak_rss_bytes"),
                pl.col("avg_rss_bytes").mean().alias("avg_rss_bytes"),
            )
        )
        mem_out = results_dir / "memory_summary.csv"
        mem.sort(["dataset_norm", "mode", "dl_mode", "seqlen", "batch_size"]).write_csv(mem_out)
        print(f"WROTE {mem_out} ({mem.height} rows)")

        if mem.height:
            mpdf = mem.with_columns(peak_gib=pl.col("peak_rss_bytes") / 2**30).to_pandas()
            g = sns.relplot(
                data=mpdf, x="batch_size", y="peak_gib", hue="dataset_norm", style="dl_mode",
                col="seqlen", col_wrap=2, kind="line", marker="o",
                facet_kws={"sharex": False, "sharey": False},
            )
            for ax in g.axes.flat:
                ax.set_xscale("log", base=2)
            g.set_axis_labels("batch_size", "peak RSS (GiB)")
            g.savefig(fig_dir / "gvl027_peak_rss.png", dpi=150, bbox_inches="tight")
            print(f"WROTE {fig_dir / 'gvl027_peak_rss.png'}")
        else:
            print("No valid memory rows after drop_nulls — skipping memory plot.")
    else:
        print("No memory CSVs found under haps_memory/ or tracks_memory/ — skipping memory report.")


if __name__ == "__main__":
    cyclopts.run(main)
