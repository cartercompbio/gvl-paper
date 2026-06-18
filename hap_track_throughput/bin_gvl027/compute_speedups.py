#! /usr/bin/env python
"""Join re-measured baselines to GVL 0.27 throughput; emit speedups + A100 check.

speedup(mode, dataset, seqlen) = max_over_grid(GVL MiB/s) / max_over_grid(baseline MiB/s).
FASTA baseline is dataset-independent (one curve); compared to each haps dataset.
pyBigWig baseline is tcga-atac, compared to GVL tracks.
"""
from pathlib import Path

from cyclopts import run


def main(
    results_dir: Path = Path("results_gvl027"),
    a100_pcie_gb_s: float = 25.0,
):
    import polars as pl

    def best(glob, group):
        paths = list(results_dir.glob(glob))
        if not paths:
            raise FileNotFoundError(f"No files matched {results_dir}/{glob} — run Task 8 first")
        frames = [pl.read_csv(p, null_values=["nan"]) for p in paths]
        df = pl.concat(frames).with_columns(
            pl.col("throughput (MiB/s)").fill_nan(None)
        ).drop_nulls("throughput (MiB/s)")
        return df.group_by(group).agg(pl.col("throughput (MiB/s)").max().alias("best_mib_s"))

    gvl_haps = best("haps/*_none.csv", ["dataset", "seqlen"])
    gvl_trk = best("tracks/*_none.csv", ["dataset", "seqlen"])
    base_fa = best("baselines/fasta.csv", ["seqlen"]).rename({"best_mib_s": "fasta_mib_s"})
    base_bw = best("baselines/pybigwig.csv", ["seqlen"]).rename({"best_mib_s": "pybigwig_mib_s"})

    haps = gvl_haps.join(base_fa, on="seqlen").with_columns(
        (pl.col("best_mib_s") / pl.col("fasta_mib_s")).alias("speedup"),
        pl.lit("haps").alias("mode"),
    )
    trk = gvl_trk.join(base_bw, on="seqlen").with_columns(
        (pl.col("best_mib_s") / pl.col("pybigwig_mib_s")).alias("speedup"),
        pl.lit("tracks").alias("mode"),
    )
    out = pl.concat([haps.select("mode", "dataset", "seqlen", "best_mib_s", "speedup"),
                     trk.select("mode", "dataset", "seqlen", "best_mib_s", "speedup")])
    # A100 check: GVL GB/s vs PCIe bandwidth (MiB/s -> GB/s)
    out = out.with_columns(
        (pl.col("best_mib_s") * 2**20 / 1e9).alias("gvl_gb_s"),
    ).with_columns(
        (pl.col("gvl_gb_s") >= a100_pcie_gb_s).alias("exceeds_a100_pcie"),
    )
    out = out.sort("mode", "dataset", "seqlen")
    out.write_csv(results_dir / "speedups.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(out)
    print(f"\nspeedup range: {out['speedup'].min():.0f}x – {out['speedup'].max():.0f}x")
    print(f"any cell exceeds A100 PCIe ({a100_pcie_gb_s} GB/s)? {out['exceeds_a100_pcie'].any()}")


if __name__ == "__main__":
    run(main)
