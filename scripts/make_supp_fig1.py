#! /usr/bin/env python3
"""Assemble Supplementary Figure 1 (A: haplotype throughput, B: track throughput).

Stacks the two panels produced by scripts/plot.py (figures/hap_throughput.png and
figures/track_throughput.png, both on the GVL 0.27.0 eager grid with the
same-hardware multi-threaded FASTA/pyBigWig baselines) into a single labeled
A/B figure, and writes it to text/images/supplement_image1.png for the
supplement. Re-run after plot.py whenever the underlying data changes.
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

proj_dir = Path(__file__).resolve().parent.parent
fig_dir = proj_dir / "figures"
out_paths = [
    fig_dir / "supp_fig1_throughput.png",
    proj_dir / "text" / "images" / "supplement_image1.png",
]

panels = [fig_dir / "hap_throughput.png", fig_dir / "track_throughput.png"]
imgs = [mpimg.imread(p) for p in panels]

# Height ratios from each panel's native aspect so neither is distorted.
fig_w = 7.0
heights = [fig_w * im.shape[0] / im.shape[1] for im in imgs]

fig, axes = plt.subplots(
    2, 1, figsize=(fig_w, sum(heights)), gridspec_kw={"height_ratios": heights}
)
for ax, im, label in zip(axes, imgs, ["A", "B"]):
    ax.imshow(im)
    ax.axis("off")
    ax.text(
        -0.01,
        1.0,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="right",
    )

fig.subplots_adjust(left=0.04, right=1.0, top=1.0, bottom=0.0, hspace=0.02)
for out in out_paths:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
