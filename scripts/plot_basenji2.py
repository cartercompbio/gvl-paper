#! /usr/bin/env python3

from pathlib import Path

import cyclopts


def main(
    genes: Path,
    sample_ids: Path,
    expr: Path,
    var_samples: Path,
    targets: Path,
    preds: Path,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import scipy.stats as st
    import seaborn as sns

    proj_dir = Path(__file__).parent.parent
    fig_dir = proj_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_regions = 3259
    n_samples = 462

    _genes = pl.read_csv(
        genes,
        has_header=False,
        new_columns=["gene_id", "chrom", "chromStart", "hgnc", "strand"],
    )
    _sample_ids = (
        pl.scan_csv(
            sample_ids,
        )
        .select("path", "sample")
        .with_columns(pl.col("path").str.extract(r"(ERR\d{6})"))
        .collect()
    )
    _expr = (
        pl.read_csv(
            expr,
            separator="\t",
            comment_prefix="#",
        )
        .rename({"TargetID": "gene_id"})
        .drop("Gene_Symbol", "Chr", "Coord")
        .with_columns(
            pl.col("gene_id").str.split(".").list.get(0),
            (cs.numeric() / cs.numeric().sum() * int(1e6)).log1p()
            / np.log(2),  # log2(CPM + 1)
        )
        .join(_genes, "gene_id")
        .sort("gene_id")
        .rename(dict(zip(_sample_ids["path"], _sample_ids["sample"])), strict=False)
    )
    _var_samples = pl.read_csv(
        var_samples,
        separator="\t",
    )["#IID"].to_list()
    samples = sorted(list(set(_var_samples).intersection(_expr.columns)))
    _expr = _expr.select("gene_id", *samples)
    assert _genes.height >= _expr.height
    assert len(samples) == _expr.width - 1

    _targets = (
        pl.read_csv(targets, separator="\t")
        .filter(pl.col("description").str.contains(r"lymphoblastoid"))["index"]
        .to_list()
    )
    basenji2_out_len = 896
    ploidy = 2
    # (r s p t)
    _pred_expr = np.memmap(
        preds,
        dtype=np.float32,
        mode="r",
        shape=(n_regions, n_samples, ploidy, len(_targets), basenji2_out_len),
    )
    # (g s p t l) -> (g s)
    huang_expr = _pred_expr[
        ..., basenji2_out_len // 2 - 5 : basenji2_out_len // 2 + 5
    ].mean(axis=(2, 3, 4))
    gene_rho = np.diag(
        st.spearmanr(_expr[:, 1:].to_numpy(), huang_expr, axis=0).statistic,  # pyright: ignore
        n_samples,
    )
    indiv_rho = np.diag(
        st.spearmanr(_expr[:, 1:].to_numpy(), huang_expr, axis=1).statistic,  # pyright: ignore
        n_regions,
    )

    # %%
    fig, ax = plt.subplots()
    sns.ecdfplot(gene_rho.ravel(), label=r"$\rho$ across genes", ax=ax, linewidth=3)
    ax.axvline(np.nanmean(gene_rho), c="k", ls="--", alpha=0.5, linewidth=3)
    sns.ecdfplot(
        indiv_rho.ravel(), label=r"$\rho$ across individuals", ax=ax, linewidth=3
    )
    ax.axvline(np.nanmean(indiv_rho), c="k", ls="--", alpha=0.5, linewidth=3)
    ax.legend(loc="best")
    fig.savefig(fig_dir / "basenji2_rho.png", dpi=300)
    fig.savefig(fig_dir / "basenji2_rho.svg")


if __name__ == "__main__":
    cyclopts.run(main)
