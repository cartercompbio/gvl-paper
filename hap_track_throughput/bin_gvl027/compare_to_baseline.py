#! /usr/bin/env python
"""Compare the GVL 0.27.0 full-bench results to the v0.6.1 baseline.

Run in the **default** pixi env (`pixi run -e default ...`): this reads CSVs and
plots with matplotlib/seaborn (absent from bench027) and never imports GVL.

Reads the Nextflow output layout under results_gvl027/:
    haps/{dataset}_{length}_{backend}_{dl_mode}.csv          (throughput)
    tracks/{dataset}_{length}_{backend}_{dl_mode}.csv        (throughput)
    haps_memory/...   tracks_memory/...                      (RSS-vs-time growth)

Only the eager `mode=none` dataloader is benchmarked (buffered was dropped — its
super-batch slicing made a short measurement window clock slice-handoff, >1 TB/s;
see CLAUDE.md). The memory pass emits an RSS-vs-time growth curve at a single
operating point (largest batch), not a peak/avg sweep.

Outputs:
  results_gvl027/parity_summary.csv  — eager throughput joined to the 0.6.1
       baseline on (dataset,mode,threads,seqlen,batch_size); ratio = v027/v061.
       INTERNAL sanity check, not a paper deliverable. 0.27.0 runs on cn-03, the
       0.6.1 baseline on faster hardware, so ratio < 1 is expected.
  results_gvl027/memory_growth_summary.csv  — per (dataset,mode,seqlen,batch_size)
       start/end/peak RSS + duration of the growth curve, absolute (no baseline).
  figures/gvl027_parity.png          — v061-vs-v027 eager-throughput scatter.
  figures/gvl027_mem_growth.png      — RSS vs elapsed time, faceted by seqlen.
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
    import matplotlib.lines as mlines
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

        # internal parity: eager mode=none (the only benchmarked dl_mode) vs the
        # 0.6.1 baseline. Both are real per-batch decodes (0.6.1 had no buffered
        # path), so this is apples-to-apples; 0.27.0 on cn-03 vs 0.6.1 on faster
        # hardware means ratio < 1 is expected and not a regression signal per se.
        eager = tput.filter(pl.col("dl_mode") == "none")
        joined = eager.join(
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
    else:
        print("No throughput CSVs found under haps/ or tracks/ — skipping throughput report.")

    # ---------- memory (RSS-vs-time growth) ----------
    # The memory pass emits one row per RSS sample (schema: ...,elapsed_ns,
    # rss_bytes) at a single operating point per (dataset,seqlen), not a
    # peak/avg sweep. Summarize start/end/peak per curve; plot RSS vs time.
    mem_frames = [
        f
        for f in (_load_dir(results_dir, d) for d in ("haps_memory", "tracks_memory"))
        if f is not None
    ]
    if mem_frames:
        mem = pl.concat(mem_frames, how="vertical_relaxed").with_columns(
            pl.col("rss_bytes").cast(pl.Float64, strict=False),
            pl.col("elapsed_ns").cast(pl.Float64, strict=False),
        ).drop_nulls("rss_bytes")

        summary = (
            mem.sort("elapsed_ns")
            .group_by(["dataset_norm", "mode", "seqlen", "batch_size"])
            .agg(
                pl.col("rss_bytes").first().alias("rss_start_bytes"),
                pl.col("rss_bytes").last().alias("rss_end_bytes"),
                pl.col("rss_bytes").max().alias("rss_peak_bytes"),
                (pl.col("elapsed_ns").max() / 1e9).alias("duration_s"),
                pl.len().alias("n_samples"),
            )
            .with_columns(
                growth_bytes=pl.col("rss_peak_bytes") - pl.col("rss_start_bytes")
            )
            .sort(["dataset_norm", "mode", "seqlen"])
        )
        mem_out = results_dir / "memory_growth_summary.csv"
        summary.write_csv(mem_out)
        print(f"WROTE {mem_out} ({summary.height} curves)")
        print(
            summary.with_columns(
                start_gib=(pl.col("rss_start_bytes") / 2**30).round(2),
                peak_gib=(pl.col("rss_peak_bytes") / 2**30).round(2),
            ).select("dataset_norm", "mode", "seqlen", "batch_size", "start_gib", "peak_gib", "duration_s")
        )

        # Per-(dataset, mode, seqlen) operating-point stats from the throughput
        # CSVs, used to annotate the growth curves. Keyed by *dataset* (not shared
        # across datasets) so each reference line matches its own curve's hue — a
        # shared (mode, seqlen) line drew a 1kGP-sized batch over the much smaller
        # TCGA curve, making TCGA's RSS look "smaller than one batch" when really
        # the line belonged to a different dataset. We compute:
        #   batch_gib     — bytes in one batch at the largest batch (the memory
        #                   operating point) = total_bytes / n_batches_measured.
        #   inst_streamed — instances actually decoded in the growth window
        #                   (throughput * duration / bytes-per-instance). This is
        #                   the key to the figure: a larger-seqlen panel can show
        #                   *less* RSS growth simply because, in the fixed window,
        #                   far fewer instances fault in from the mmap'd store.
        tput_frames_m = [
            f for f in (_load_dir(results_dir, k) for k in ("haps", "tracks")) if f is not None
        ]
        stats_d: dict[tuple[str, str, int], dict] = {}
        if tput_frames_m:
            tput_ok = pl.concat(tput_frames_m, how="vertical_relaxed").with_columns(
                pl.col("throughput (MiB/s)").cast(pl.Float64, strict=False)
            ).filter(pl.col("throughput (MiB/s)") > 0)
            if not tput_ok.is_empty():
                # bytes/instance = max over the grid of (bytes/batch)/batch_size;
                # partial (epoch-boundary) batches only undercount, so max is true.
                bpi = (
                    tput_ok.with_columns(
                        bpi=pl.col("total_bytes")
                        / pl.col("n_batches_measured")
                        / pl.col("batch_size")
                    )
                    .group_by(["dataset_norm", "mode", "seqlen"])
                    .agg(pl.col("bpi").max())
                )
                op = (
                    tput_ok.sort("batch_size", descending=True)
                    .group_by(["dataset_norm", "mode", "seqlen"])
                    .agg(
                        (pl.col("total_bytes") / pl.col("n_batches_measured"))
                        .first()
                        .alias("batch_bytes"),
                        pl.col("throughput (MiB/s)")
                        .filter(pl.col("batch_size") == pl.col("batch_size").max())
                        .median()
                        .alias("mib_s"),
                    )
                )
                stats = (
                    op.join(bpi, on=["dataset_norm", "mode", "seqlen"])
                    .join(
                        summary.select(["dataset_norm", "mode", "seqlen", "duration_s"]),
                        on=["dataset_norm", "mode", "seqlen"],
                        how="left",
                    )
                    .with_columns(
                        batch_gib=pl.col("batch_bytes") / 2**30,
                        inst_streamed=(pl.col("mib_s") * 2**20 * pl.col("duration_s"))
                        / pl.col("bpi"),
                    )
                )
                stats_d = {
                    (r["dataset_norm"], r["mode"], int(r["seqlen"])): r
                    for r in stats.iter_rows(named=True)
                }
        JOB_MEM_GIB = 96.0  # BENCH_HAPS/TRACKS `memory 96.GB` request

        # Per-curve decode rate (instances / s) so the x-axis can be normalized
        # from wall-clock to *cumulative instances decoded* (dataset coverage).
        # Coverage is the right normalizer here: the grid scales batch_size
        # inversely with seqlen, so "number of batches" would mean wildly
        # different coverage across panels, while instances-decoded is directly
        # comparable. (True "epochs" would need each dataset's total instance
        # count, which isn't recorded in any throughput/memory CSV.) Rate is
        # taken as constant = inst_streamed / duration; throughput sawtooth makes
        # this approximate, but it's representative of the steady-state decode.
        rate_df = None
        rate_rows = [
            {
                "dataset_norm": k[0], "mode": k[1], "seqlen": k[2],
                "inst_rate": r["inst_streamed"] / r["duration_s"],
            }
            for k, r in stats_d.items()
            if r.get("inst_streamed") is not None and r.get("duration_s")
        ]
        if rate_rows:
            rate_df = pl.DataFrame(rate_rows)

        if mem.height:
            mem_x = (
                mem.join(rate_df, on=["dataset_norm", "mode", "seqlen"], how="left")
                if rate_df is not None
                else mem.with_columns(inst_rate=pl.lit(None, dtype=pl.Float64))
            )
            mpdf = mem_x.with_columns(
                elapsed_s=pl.col("elapsed_ns") / 1e9,
                rss_gib=pl.col("rss_bytes") / 2**30,
                inst_decoded_m=(pl.col("elapsed_ns") / 1e9 * pl.col("inst_rate")) / 1e6,
            ).to_pandas()
            # Map raw keys to display names for the legend.
            DATASET_LABELS = {"1kgp": "1000 Genomes", "tcga-atac": "TCGA ATAC"}
            MODE_LABELS = {"haps": "Haplotypes", "tracks": "Tracks"}
            mpdf["Dataset"] = mpdf["dataset_norm"].map(DATASET_LABELS).fillna(mpdf["dataset_norm"])
            mpdf["Mode"] = mpdf["mode"].map(MODE_LABELS).fillna(mpdf["mode"])
            # Fixed hue order + palette so reference lines can be colored to match
            # their dataset's curve.
            hue_order = sorted(mpdf["Dataset"].unique())
            palette = dict(zip(hue_order, sns.color_palette(n_colors=len(hue_order))))
            # x = instances decoded if we have rates for every curve, else fall
            # back to wall-clock so the figure still renders.
            x_col = "inst_decoded_m" if mpdf["inst_decoded_m"].notna().all() else "elapsed_s"
            x_label = "instances decoded (M)" if x_col == "inst_decoded_m" else "elapsed (s)"
            sns.set_context("notebook", font_scale=2.0)  # 2x larger fonts
            g = sns.relplot(
                data=mpdf, x=x_col, y="rss_gib", hue="Dataset", style="Mode",
                hue_order=hue_order, palette=palette,
                col="seqlen", col_wrap=2, kind="line", estimator=None,
                linewidth=1,
                facet_kws={"sharex": False, "sharey": False},
            )
            g.set_axis_labels(x_label, "RSS (GiB)")
            for ax in g.axes.flat:
                # Only reference line kept: the 96 GiB job request (allocated RAM).
                # RSS climbs above it because the excess is reclaimable file-backed
                # mmap page cache, not anonymous heap.
                ax.axhline(JOB_MEM_GIB, c="r", ls=":", lw=2.5, alpha=0.8)
            g.tight_layout()
            # Append the allocated-RAM reference line to the legend, set off from
            # the Dataset/Mode entries by a blank spacer row.
            if g._legend is not None:
                handles = list(g._legend.legend_handles)
                labels = [t.get_text() for t in g._legend.texts]
                spacer = mlines.Line2D([], [], color="none")
                ram_line = mlines.Line2D([], [], color="r", ls=":", lw=2.5, alpha=0.8)
                handles += [spacer, ram_line]
                labels += ["", "Alloc. RAM"]
                g._legend.remove()
                g.figure.legend(handles, labels, loc="center right", frameon=False)
            for ext in ("png", "svg"):
                out = fig_dir / f"gvl027_mem_growth.{ext}"
                g.savefig(out, dpi=150, bbox_inches="tight")
                print(f"WROTE {out}")
        else:
            print("No valid memory samples after drop_nulls — skipping memory plot.")
    else:
        print("No memory CSVs found under haps_memory/ or tracks_memory/ — skipping memory report.")


if __name__ == "__main__":
    cyclopts.run(main)
