#! /usr/bin/env python3

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401

# %%
proj_dir = Path(__file__).parent.parent
fig_dir = proj_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
sns.set_context("notebook", font_scale=1.5)

vars_per_block = 4
n_variants = 3 * vars_per_block
n_samples = 6
n_blocks = -(-n_variants // vars_per_block)
block_width = 1
rows = n_variants + n_blocks * block_width - 1

grid = np.zeros((rows, n_samples), np.uint32)

rng = np.random.default_rng(1)
n_queries = 6
min_len = 5
max_len = rows // 2
starts = rng.integers(0, rows, n_queries)
ends = (starts + rng.integers(min_len, max_len + 1, n_queries)).clip(max=rows - 1)
samples = rng.choice(n_samples, n_queries, replace=False)

i = 0
block_markers = np.lib.stride_tricks.sliding_window_view(
    grid, (block_width, n_samples), writeable=True
)[vars_per_block :: vars_per_block + block_width]
block_markers[:] = n_queries + 4

for i, (s, e, sp) in enumerate(zip(starts, ends, samples)):
    grid[s:e, sp] = i + 3

fig, ax = plt.subplots()
img = ax.imshow(
    grid,
    aspect="auto",
    interpolation="none",
    cmap=sns.color_palette("cubehelix", as_cmap=True),
    vmin=0,
    vmax=n_queries + 4,
)
ax.set(xticks=[], yticks=[], xlabel="Samples", ylabel="Variants")
ax.xaxis.set(label_position="top")
sns.despine(ax=ax, left=True, bottom=True)
fig.tight_layout()
fig.savefig(fig_dir / "memory_blocks.png", dpi=150)
fig.savefig(fig_dir / "memory_blocks.svg")


block_markers[:] = 0
for i, (s, e, sp) in enumerate(zip(starts, ends, samples)):
    grid[s:e, sp] = i + 3
x = np.stack([grid.ravel(), grid.T.ravel()], axis=0)
fig, ax = plt.subplots()
ax.imshow(
    x,
    aspect="auto",
    interpolation="none",
    cmap=sns.color_palette("cubehelix", as_cmap=True),
    vmin=0,
    vmax=n_queries + 4,
)
sns.despine(ax=ax, left=True, bottom=True)
ax.set(xticks=[], yticks=[0, 1], yticklabels=["Variant-major", "Sample-major"])
fig.tight_layout()
fig.savefig(fig_dir / "memory_layouts.png", dpi=150)
fig.savefig(fig_dir / "memory_layouts.svg")
