from pathlib import Path

METHOD_LABELS = {
    "svar": "SVAR",
    "bcf": "BCF",
    "pgen": "PGEN",
    "presubset_bcf": "PRESUB-BCF",
}


def lmplot(df, x_col: str, y_col: str, x_label: str, y_label: str, hue_order: list[str], output_dir: Path, stem: str):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    g = sns.lmplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="method_label",
        hue_order=hue_order,
        lowess=True,
    )

    xticks = np.arange(
        np.floor(df[x_col].min()),
        np.ceil(df[x_col].max()) + 1,
    )
    g.ax.set_xticks(xticks)
    g.ax.set_xticklabels([f"$10^{{{int(x)}}}$" for x in xticks])

    yticks = np.arange(
        np.floor(df[y_col].min()),
        np.ceil(df[y_col].max()) + 1,
    )
    g.ax.set_yticks(yticks)
    g.ax.set_yticklabels([f"$10^{{{int(y)}}}$" for y in yticks])

    g.ax.set_xlabel(x_label)
    g.ax.set_ylabel(y_label)
    g.legend.set_title("Method")

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "svg", "pdf"):
        g.savefig(output_dir / f"{stem}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close()
