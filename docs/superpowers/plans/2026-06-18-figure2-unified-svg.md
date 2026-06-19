# Unified Figure 2 (.svg) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `figures/figure2.svg` — a single vector figure containing manuscript Figure 2's data-driven panels (A disk, B variant-vs-N, C haplotype throughput, D track throughput, F Basenji2 ρ), with a reserved labeled slot for panel E (GPU screenshot, composited later).

**Architecture:** ultraplot owns the 2×3 figure (layout, rc, `a)–f)` lettering); seaborn axes-level functions draw each panel onto an ultraplot axes. Shared data loaders move into a new `scripts/_fig_data.py` imported by both the existing `plot.py` and the new `plot_figure2.py`. Panel F reads a small ρ cache written by `plot_basenji2.py` so the composite needs no controlled-access data.

**Tech Stack:** Python 3.12, pixi (`default` env), polars, seaborn, ultraplot (new dep), matplotlib.

**Spec:** `docs/superpowers/specs/2026-06-18-figure2-unified-svg-design.md`

**Note on verification:** This is figure code. "Tests" are: the script imports/runs without error, the expected output files appear, and the rendered PNG is visually inspected. There is no pytest suite for these scripts; do not invent one.

---

## File Structure

- **Create** `scripts/_fig_data.py` — shared GVL-0.27 data loaders + constants (`gvl027_peak`, `gvl027_grid`, `disk_usage_df`, `variant_n_df`, `GVL027_LABELS`, `RAM_BW_GBPS`, `RAM_BW_LABEL`, `METHOD_LABELS`).
- **Modify** `scripts/plot.py` — import the moved helpers instead of defining them; behavior unchanged.
- **Modify** `scripts/plot_basenji2.py` — additionally write `figures/basenji2_rho.npz`.
- **Create** `scripts/plot_figure2.py` — panel functions + ultraplot assembly → `figures/figure2.{svg,png}`.
- **Modify** `pixi.toml` — add `ultraplot` dependency.

---

## Task 0: Add the ultraplot dependency

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Add ultraplot to the default/bench feature**

Run (prefer conda-forge; pixi resolves it into the default env):

```bash
pixi add ultraplot
```

If conda-forge resolution fails, fall back to PyPI:

```bash
pixi add --pypi ultraplot
```

- [ ] **Step 2: Verify it imports in the default env**

Run:

```bash
pixi r python -c "import ultraplot as uplt; print('ultraplot', uplt.__version__)"
```

Expected: prints `ultraplot <version>` (>= 2.x) with no ImportError.

- [ ] **Step 3: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "build: add ultraplot for unified figure assembly"
```

---

## Task 1: Extract shared data loaders into `scripts/_fig_data.py`

**Files:**
- Create: `scripts/_fig_data.py`
- Modify: `scripts/plot.py` (remove the moved defs at lines ~26–67 and the `memory` literal at ~285–309; add an import)

- [ ] **Step 1: Create `scripts/_fig_data.py`**

```python
"""Shared data-loading helpers + constants for the GVL 0.27 throughput figures.

Imported by scripts/plot.py (standalone panels) and scripts/plot_figure2.py
(the unified Figure 2 composite). All throughput numbers come from the GVL
0.27.0 eager (mode=none) bench in results_gvl027/; see root CLAUDE.md.
"""

import glob
from pathlib import Path

import polars as pl

proj_dir = Path(__file__).resolve().parent.parent

RAM_BW_GBPS = 35.0
RAM_BW_LABEL = "cn-03 max RAM\nbandwidth"
GVL027_LABELS = {
    "TCGA_ATAC": "GVL: TCGA BRCA ATAC (n=62)",
    "1KGP": "GVL: 1000 Genomes (n=3,202)",
    "UKBB": "GVL: Biobank (n=487,409)",
}
# Variant file-type labels (mirrors variant_throughput/bin/_plot_common.py).
METHOD_LABELS = {
    "svar": "SVAR",
    "bcf": "BCF",
    "pgen": "PGEN",
    "presubset_bcf": "PRESUB-BCF",
}


def gvl027_peak(result_glob: str) -> pl.DataFrame:
    """Max eager (mode=none) throughput per (dataset, seqlen), in GB/s."""
    files = sorted(glob.glob(str(proj_dir / result_glob)))
    if not files:
        raise FileNotFoundError(f"no GVL 0.27.0 result CSVs matched {result_glob!r}")
    df = pl.concat([pl.read_csv(c) for c in files], how="vertical_relaxed").with_columns(
        pl.col("throughput (MiB/s)").cast(pl.Float64, strict=False)
    )
    df = df.filter(
        pl.col("throughput (MiB/s)").is_finite() & (pl.col("throughput (MiB/s)") > 0)
    )
    return (
        df.group_by("dataset", "seqlen")
        .agg(throughput=(pl.col("throughput (MiB/s)").max() * 2**20 / 1e9))
        .sort("dataset", "seqlen")
    )


def gvl027_grid(result_glob: str) -> pl.DataFrame:
    """Per-cell eager (mode=none) throughput grid, GB/s, with n_nucleotides."""
    files = sorted(glob.glob(str(proj_dir / result_glob)))
    if not files:
        raise FileNotFoundError(f"no GVL 0.27.0 result CSVs matched {result_glob!r}")
    df = pl.concat([pl.read_csv(c) for c in files], how="vertical_relaxed").with_columns(
        pl.col("throughput (MiB/s)").cast(pl.Float64, strict=False)
    )
    df = df.filter(
        pl.col("throughput (MiB/s)").is_finite() & (pl.col("throughput (MiB/s)") > 0)
    )
    return df.with_columns(
        n_nucleotides=pl.col("seqlen") * pl.col("batch_size"),
        throughput=pl.col("throughput (MiB/s)") * 2**20 / 1e9,  # GB/s
    )


def disk_usage_df() -> pl.DataFrame:
    """Personalized-genome disk footprint, GVL vs compressed FASTA (Fig. 2A)."""
    compressed_hg37 = 0.987
    compressed_hg38 = 0.875
    return pl.DataFrame({
        "Dataset": [
            "TCGA BRCA ATAC (n=62)",
            "TCGA BRCA ATAC (n=62)",
            "1000 Genomes (n=3,202)",
            "1000 Genomes (n=3,202)",
            "GDC (n=16,007)",
            "GDC (n=16,007)",
            "Biobank, chr22 (n=487,409)",
            "Biobank, chr22 (n=487,409)",
        ],
        "Implementation": ["GVL", "FASTA"] * 4,
        "Disk Space (GB)": [
            0.173,
            compressed_hg37 * 62 * 2,
            3.1,
            compressed_hg37 * 3202 * 2,
            7.9,
            compressed_hg38 * 16007,
            30,
            0.0096 * 487409 * 2,  # just chr22
        ],
    })


def variant_n_df() -> "pl.DataFrame":
    """Variant calls/sec vs cohort size N, by file type (Fig. 2B, the N-sweep).

    Reads variant_throughput/results/*_throughput.csv. The N-sweep query
    length(s) are those whose rows span more than one distinct n_samples
    (matches variant_throughput/bin/plot_throughput.py). Returns columns:
    log10_n_samples, log10_calls_per_sec, method_label, calls_per_sec.
    """
    csvs = sorted(glob.glob(str(proj_dir / "variant_throughput/results/*_throughput.csv")))
    if not csvs:
        raise FileNotFoundError("no variant_throughput/results/*_throughput.csv found")
    raw = pl.concat(
        [pl.read_csv(p, schema_overrides={"setup_ns": pl.Int64}) for p in csvs],
        how="vertical_relaxed",
    )
    n_sweep_qlens = (
        raw.group_by("query_length")
        .agg(pl.col("n_samples").n_unique().alias("n_unique"))
        .filter(pl.col("n_unique") > 1)["query_length"]
        .to_list()
    )
    return (
        raw.filter(pl.col("query_length").is_in(n_sweep_qlens))
        .filter((pl.col("n_calls") > 0) & (pl.col("elapsed_ns") > 0))
        .with_columns(
            (pl.col("n_calls") / (pl.col("elapsed_ns") * 1e-9)).alias("calls_per_sec")
        )
        .with_columns(
            pl.col("n_samples").log(base=10).alias("log10_n_samples"),
            pl.col("calls_per_sec").log(base=10).alias("log10_calls_per_sec"),
            pl.col("method").replace(METHOD_LABELS).alias("method_label"),
        )
    )
```

- [ ] **Step 2: Verify the new module loads all panels' data**

Run:

```bash
pixi r python -c "
import sys; sys.path.insert(0, 'scripts')
import _fig_data as fd
print('haps', fd.gvl027_peak('results_gvl027/haps/*_none.csv').shape)
print('tracks', fd.gvl027_peak('results_gvl027/tracks/*_none.csv').shape)
print('fasta', fd.gvl027_peak('results_gvl027/baselines/fasta.csv').shape)
print('bigwig', fd.gvl027_peak('results_gvl027/baselines/pybigwig.csv').shape)
print('disk', fd.disk_usage_df().shape)
print('variant_n', fd.variant_n_df().shape, sorted(fd.variant_n_df()['method_label'].unique().to_list()))
"
```

Expected: each prints a non-empty shape with no exception; `variant_n` method labels include `['BCF', 'PGEN', 'PRESUB-BCF', 'SVAR']`.

- [ ] **Step 3: Update `scripts/plot.py` to import the moved helpers**

In `scripts/plot.py`, delete the constant/function block (the `RAM_BW_GBPS` / `RAM_BW_LABEL` / `GVL027_LABELS` definitions and the `gvl027_peak` / `gvl027_grid` function defs, currently lines ~26–67) and add, just after the existing imports near the top:

```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_data import (  # noqa: E402
    GVL027_LABELS,
    RAM_BW_GBPS,
    RAM_BW_LABEL,
    disk_usage_df,
    gvl027_grid,
    gvl027_peak,
)
```

Then replace the inline `memory = pl.DataFrame({...})` literal (the disk-usage block, currently lines ~285–309) with:

```python
memory = disk_usage_df()
```

Leave everything else in `plot.py` unchanged.

- [ ] **Step 4: Verify `plot.py` still runs and regenerates a standalone panel**

Run:

```bash
pixi r scripts/plot.py && ls -la figures/best_haplotype_performance.svg figures/disk_usage.svg
```

Expected: runs to completion (it prints the variant ratio table at the end), and both SVGs exist with a fresh timestamp.

- [ ] **Step 5: Commit**

```bash
git add scripts/_fig_data.py scripts/plot.py
git commit -m "refactor: extract shared figure data loaders into _fig_data"
```

---

## Task 2: Cache Basenji2 ρ arrays for the composite

**Files:**
- Modify: `scripts/plot_basenji2.py` (add an `np.savez` after `indiv_rho` is computed, ~line 93)

- [ ] **Step 1: Add the cache write**

In `scripts/plot_basenji2.py`, immediately after the `indiv_rho = np.diag(...)` block (currently ending ~line 93, before the `# %%` that starts plotting), add:

```python
    np.savez(fig_dir / "basenji2_rho.npz", gene_rho=gene_rho, indiv_rho=indiv_rho)
```

(Indentation: inside `main`, same level as the surrounding code.)

- [ ] **Step 2: Run the basenji2 script to produce the cache**

Run (same invocation as `run_scripts.sh`; requires the controlled-access RNA-seq at `/carter/users/dlaub/data/1kGP-rna-seq`):

```bash
pixi r scripts/plot_basenji2.py \
    basenji2/gene_list.csv \
    /carter/users/dlaub/data/1kGP-rna-seq/sample_id_to_bigwig.csv \
    /carter/users/dlaub/data/1kGP-rna-seq/E-GEUV-3_analysis/rna/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz \
    /carter/users/dlaub/data/1kGP-rna-seq/E-GEUV-3_analysis/geuvadis.psam \
    basenji2/targets_human.tsv \
    basenji2/data/preds_hg19.npy
```

Expected: completes and writes `figures/basenji2_rho.npz` (plus the existing `basenji2_rho.svg/png`).

- [ ] **Step 3: Verify the cache is well-formed**

Run:

```bash
pixi r python -c "
import numpy as np
d = np.load('figures/basenji2_rho.npz')
print('gene_rho', d['gene_rho'].shape, 'mean', float(np.nanmean(d['gene_rho'])))
print('indiv_rho', d['indiv_rho'].shape, 'mean', float(np.nanmean(d['indiv_rho'])))
"
```

Expected: `gene_rho` mean near 0.5, `indiv_rho` mean near 0 (matches the manuscript: ρ≈0.5 across genes, ≈0 across individuals).

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_basenji2.py
git commit -m "feat: cache basenji2 rho arrays to npz for figure assembly"
```

---

## Task 3: Build the unified figure `scripts/plot_figure2.py`

**Files:**
- Create: `scripts/plot_figure2.py`

- [ ] **Step 1: Write the composite script**

```python
#! /usr/bin/env python3
"""Assemble manuscript Figure 2 data panels into one vector figures/figure2.svg.

Panels: a) disk usage, b) variant calls/sec vs cohort N, c) haplotype
throughput, d) track throughput, e) reserved GPU-utilization slot (composited
separately), f) Basenji2 rho ECDF. ultraplot owns layout + a)-f) lettering;
seaborn axes-level functions draw each panel. See
docs/superpowers/specs/2026-06-18-figure2-unified-svg-design.md.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import ultraplot as uplt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_data import (  # noqa: E402
    GVL027_LABELS,
    RAM_BW_GBPS,
    RAM_BW_LABEL,
    disk_usage_df,
    gvl027_peak,
    variant_n_df,
)

proj_dir = Path(__file__).resolve().parent.parent
fig_dir = proj_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

HAP_HUE_ORDER = [
    "GVL: TCGA BRCA ATAC (n=62)",
    "GVL: 1000 Genomes (n=3,202)",
    "GVL: Biobank (n=487,409)",
    "FASTA",
]
VARIANT_ORDER = ["SVAR", "PRESUB-BCF", "BCF", "PGEN"]


def _ram_bw_line(ax, xmin):
    ax.axhline(RAM_BW_GBPS, c="k", ls="--", alpha=0.5, linewidth=2)
    ax.text(xmin, RAM_BW_GBPS, RAM_BW_LABEL, va="bottom", ha="left", fontsize=7)


def panel_disk(ax):
    df = disk_usage_df().to_pandas()
    sns.barplot(df, x="Disk Space (GB)", y="Dataset", hue="Implementation", ax=ax)
    ax.format(xscale="log", xlabel="Disk space (GB)", ylabel="", title="Storage")
    ax.legend(loc="lr", ncols=1)


def panel_variant_n(ax):
    df = variant_n_df().to_pandas()
    palette = dict(zip(VARIANT_ORDER, sns.color_palette(n_colors=len(VARIANT_ORDER))))
    for method in VARIANT_ORDER:
        sub = df[df["method_label"] == method]
        if sub.empty:
            continue
        sns.regplot(
            data=sub,
            x="log10_n_samples",
            y="log10_calls_per_sec",
            lowess=True,
            ax=ax,
            label=method,
            color=palette[method],
            scatter_kws=dict(s=12, alpha=0.4),
            line_kws=dict(linewidth=2),
        )
    ax.format(
        xlabel=r"$\log_{10}$ cohort size (N)",
        ylabel=r"$\log_{10}$ variant calls/s",
        title="Variant query throughput",
    )
    ax.legend(loc="lr", ncols=1)


def panel_haps(ax):
    gvl = gvl027_peak("results_gvl027/haps/*_none.csv").with_columns(
        pl.col("dataset").replace(GVL027_LABELS)
    )
    fasta = gvl027_peak("results_gvl027/baselines/fasta.csv")
    data = pl.concat([gvl, fasta], how="diagonal_relaxed").rename({"dataset": "Dataset"})
    sns.lineplot(
        data.to_pandas(),
        x="seqlen",
        y="throughput",
        hue="Dataset",
        hue_order=HAP_HUE_ORDER,
        ax=ax,
        linewidth=2.5,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    _ram_bw_line(ax, gvl["seqlen"].min())
    ax.format(
        xscale="log", yscale="log", xlabel="Sequence length",
        ylabel="Throughput (GB/s)", title="Haplotypes vs FASTA",
    )
    ax.legend(loc="lr", ncols=1, fontsize=6)


def panel_tracks(ax):
    gvl = gvl027_peak("results_gvl027/tracks/*_none.csv").sort("seqlen")
    bw = gvl027_peak("results_gvl027/baselines/pybigwig.csv").sort("seqlen")
    sns.lineplot(gvl.to_pandas(), x="seqlen", y="throughput", ax=ax, label="GVL",
                 linewidth=2.5, solid_joinstyle="round", solid_capstyle="round")
    sns.lineplot(bw.to_pandas(), x="seqlen", y="throughput", ax=ax, label="BigWig",
                 color="C2", linewidth=2.5, solid_joinstyle="round", solid_capstyle="round")
    _ram_bw_line(ax, bw["seqlen"].min())
    ax.format(
        xscale="log", yscale="log", xlabel="Sequence length",
        ylabel="Throughput (GB/s)", title="Tracks vs BigWig",
    )
    ax.legend(loc="lr", ncols=1)


def panel_gpu_placeholder(ax):
    ax.format(title="GPU utilization", xlabel="", ylabel="")
    ax.format(xticks=[], yticks=[])
    ax.text(
        0.5, 0.5, "GPU utilization\n(composited separately)",
        ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=8,
    )


def panel_basenji2(ax):
    cache = fig_dir / "basenji2_rho.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} missing — run scripts/plot_basenji2.py first "
            "(see run_scripts.sh) to generate the rho cache."
        )
    d = np.load(cache)
    gene_rho, indiv_rho = d["gene_rho"].ravel(), d["indiv_rho"].ravel()
    sns.ecdfplot(gene_rho, label=r"$\rho$ across genes", ax=ax, linewidth=2.5)
    ax.axvline(np.nanmean(gene_rho), c="k", ls="--", alpha=0.5, linewidth=2)
    sns.ecdfplot(indiv_rho, label=r"$\rho$ across individuals", ax=ax, linewidth=2.5)
    ax.axvline(np.nanmean(indiv_rho), c="k", ls="--", alpha=0.5, linewidth=2)
    ax.format(xlabel=r"Spearman $\rho$", ylabel="Proportion", title="Basenji2 evaluation")
    ax.legend(loc="ul", ncols=1, fontsize=6)


def main():
    fig, axs = uplt.subplots(nrows=2, ncols=3, refwidth=2.3, share=False)
    fig.format(abc="a)", abcloc="ul")
    panel_disk(axs[0])
    panel_variant_n(axs[1])
    panel_haps(axs[2])
    panel_tracks(axs[3])
    panel_gpu_placeholder(axs[4])
    panel_basenji2(axs[5])
    fig.save(fig_dir / "figure2.svg")
    fig.save(fig_dir / "figure2.png", dpi=200)
    print("wrote", fig_dir / "figure2.svg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the composite**

Run:

```bash
pixi r scripts/plot_figure2.py && ls -la figures/figure2.svg figures/figure2.png
```

Expected: prints `wrote .../figure2.svg`; both files exist, `figure2.svg` is a non-trivial size (> 20 KB).

- [ ] **Step 3: Visually inspect the rendered PNG**

Open `figures/figure2.png` (Read tool renders it). Confirm: 6 cells laid out 2×3 with `a)–f)` letters; a) bars, b) variant lines, c) hap curves with RAM-bw dashed line, d) track curves with RAM-bw dashed line, e) the gray placeholder box, f) two ECDF curves. Check no axis labels/legends are clipped or overlapping the panel letters.

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_figure2.py
git commit -m "feat: assemble unified Figure 2 vector svg (panels a-d, f)"
```

---

## Task 4: Polish pass (only if Step 3.3 found issues)

**Files:**
- Modify: `scripts/plot_figure2.py`

- [ ] **Step 1: Fix layout issues observed in the PNG**

Common adjustments (apply only what the inspection requires):
- Legends overlapping data → move `loc` (e.g. `"ll"`/`"ur"`) or shrink `fontsize`.
- Panel letters colliding with titles → set `fig.format(abcloc="l")` (letter to the left of the title) or `abcloc="ul"` inset.
- Panels too cramped → raise `refwidth` (e.g. `2.6`) or add `hspace`/`wspace` to `uplt.subplots(...)`.
- RAM-bw label clipped → adjust the `va`/`ha`/`fontsize` in `_ram_bw_line`.

- [ ] **Step 2: Re-run and re-inspect**

Run:

```bash
pixi r scripts/plot_figure2.py
```

Open `figures/figure2.png` and confirm the issues are resolved.

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_figure2.py
git commit -m "style: polish unified Figure 2 layout"
```

---

## Self-review notes

- **Spec coverage:** Task 0 = ultraplot prerequisite; Task 1 = `_fig_data.py` + `plot.py` refactor (panels a/b/c/d data); Task 2 = panel-F ρ cache; Task 3 = all 5 panels + reserved E slot + outputs; Task 4 = styling. Panel B = N-scaling view (`variant_n_df`), matching the spec/roadmap decision.
- **No global legend:** implemented as per-panel `ax.legend(...)`, consistent with the spec's flagged reality.
- **Naming consistency:** `gvl027_peak`, `disk_usage_df`, `variant_n_df`, `METHOD_LABELS`, `HAP_HUE_ORDER`, `VARIANT_ORDER`, `panel_*` used identically across Tasks 1 and 3.
- **Out of scope (unchanged):** panel E screenshot, Fig 1 schematic, Supp. Figs, Google-Docs/citation steps.
