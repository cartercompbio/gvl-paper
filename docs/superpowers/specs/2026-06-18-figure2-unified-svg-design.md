# Design: Unified Figure 2 as a single vector `.svg`

**Date:** 2026-06-18
**Status:** approved (design) — pending spec review
**Topic:** Assemble the data-driven panels of manuscript Figure 2 into one
publication-ready vector `figure2.svg` (replacing per-panel export + Figma
composition for these panels).

## Goal

Produce `figures/figure2.svg` (+ `figure2.png` preview) containing the
**data-driven** panels of Figure 2 — **A, B, C, D, F** — as one multi-panel
vector figure with automatic `a)–f)` lettering and consistent styling. Panel
**E** (GPU utilization, a training screenshot) and **Figure 1** (architecture
schematic) are out of scope — they remain Figma/Claude-Design composites.

This lets the user assemble most of Figure 2 directly from code instead of
exporting standalone SVGs and arranging them by hand.

## Background (verified 2026-06-18)

Manuscript Figure 2 is six panels (`text/manuscript.md` caption, line 113):

| Panel | Content | Current source |
|---|---|---|
| A | Personalized-genome disk usage, GVL vs FASTA | `scripts/plot.py` → `disk_usage.svg` (`sns.catplot`) |
| B | Variant query throughput across 1kGP cohort-size subsets, by file type | `variant_throughput` `n_plot.*` (`lmplot`); also `plot.py` `variant_throughput.svg` |
| C | Haplotype throughput, GVL vs FASTA | `plot.py` → `best_haplotype_performance.svg` (`sns.relplot`) |
| D | Track throughput, GVL vs pyBigWig | `plot.py` → `best_track_performance.svg` (`sns.relplot`) |
| E | GPU utilization during training | external screenshot — **out of scope** |
| F | Basenji2 ρ ECDF (across genes vs individuals) | `scripts/plot_basenji2.py` → `basenji2_rho.svg` |

All throughput numbers are GVL 0.27 (`results_gvl027/`); see root `CLAUDE.md`.

### Panel B decision
The variant data (`variant_throughput/results/*_throughput.csv`, schema
`dataset,method,query_length,n_samples,replicate,n_pairs,n_calls,elapsed_ns,setup_ns`)
supports two views: throughput **vs query length** (full cohort) and throughput
**vs cohort size N** (`n_plot`). Per `roadmap.md`, **2B = the N-scaling view**
(`x = log10(n_samples)`, `y = log10(calls/sec)`, hue = file type). Because it is
colored by file type (SVAR / BCF / PGEN / pre-subset BCF), this single panel
carries both the format comparison and the sample-scaling story.
N values present: {10, 32, 100, 316, 1000, 3202}. The N-sweep query length(s) are
identified as in `variant_throughput/bin/plot_throughput.py` (query lengths whose
rows span >1 distinct `n_samples`).

## Layout

ultraplot owns the figure (`uplt.subplots`), 2 rows × 3 cols, automatic
`abc="a)"`:

```
 a) Disk usage        b) Variant vs cohort N    c) Haplotype throughput
 d) Track throughput  e) [GPU util — reserved]  f) Basenji2 ρ ECDF
```

Panel **e)** is a reserved framed empty axes carrying the `e)` letter and a faint
"GPU utilization (composited separately)" placeholder, so the composite geometry
matches the final 6-panel figure and the screenshot drops into the correct slot.

## Panels (all axes-level seaborn drawn onto ultraplot axes)

| Panel | Plot | Primitive | Notes |
|---|---|---|---|
| a) | GVL vs FASTA disk, horizontal bars, log-x | `sns.barplot(ax=)` | Data block already in `plot.py` (`memory` DataFrame). |
| b) | calls/s vs cohort N, hue=file type, lowess fit | `sns.regplot(lowess=True, ax=)` per method (or `scatterplot`+lowess) | log-log; **axes-level** replacement for the figure-level `lmplot`. |
| c) | GVL (3 datasets) vs FASTA, log-log + RAM-bw line | `sns.lineplot(ax=)` | `gvl027_peak("results_gvl027/haps/*_none.csv")` + `baselines/fasta.csv`. |
| d) | GVL vs BigWig, log-log + RAM-bw line | `sns.lineplot(ax=)` | `gvl027_peak("results_gvl027/tracks/*_none.csv")` + `baselines/pybigwig.csv`. |
| f) | ECDF of ρ across genes vs individuals + mean lines | `sns.ecdfplot(ax=)` | reads cached ρ arrays (see below). |

The `RAM_BW_GBPS = 35` dashed ceiling line + label is reused in c) and d).

## Two flagged realities (confirmed with user)

1. **No single global legend.** Each panel encodes a different categorical
   variable (Implementation / File type / Dataset / GVL-vs-BigWig / ρ-type), so
   one shared legend is not meaningful. Resolution: compact per-panel legends,
   unified styling (one rc, one color logic, one RAM-bw line style), `a)–f)`
   lettering. This is the honest reading of "consolidated, not redundant."

2. **Panel F needs cached ρ arrays.** `plot_basenji2.py` recomputes ρ from
   controlled-access RNA-seq + a heavy `np.memmap` every run. Add a one-time
   cache: `plot_basenji2.py` writes `figures/basenji2_rho.npz` (two small float
   arrays: `gene_rho`, `indiv_rho`). `plot_figure2.py` reads that cache, so the
   composite rebuilds in seconds with no controlled-data dependency. Requires
   running `plot_basenji2.py` once (already part of `run_scripts.sh`). If the
   cache is missing, `plot_figure2.py` errors with a clear "run plot_basenji2.py
   first" message.

## Code structure (DRY, low-risk)

- **New module `scripts/_fig_data.py`** — move the pure shared helpers/constants
  out of `plot.py`: `gvl027_peak`, `gvl027_grid`, `GVL027_LABELS`, `RAM_BW_GBPS`,
  `RAM_BW_LABEL`. Both scripts import from it.
- **`scripts/plot.py`** — unchanged behavior; just imports the moved helpers
  instead of defining them. Continues emitting all standalone SVGs (the
  supplement still needs `hap_throughput`/`track_throughput` for Supp. Fig. 1).
- **`scripts/plot_basenji2.py`** — add the `figures/basenji2_rho.npz` cache write
  (additive; existing SVG/PNG output preserved).
- **New `scripts/plot_figure2.py`** — `panel_disk(ax)`, `panel_variant_n(ax)`,
  `panel_haps(ax)`, `panel_tracks(ax)`, `panel_basenji2(ax)`,
  `panel_gpu_placeholder(ax)`; an assembly section building the `uplt.subplots`
  grid and saving `figures/figure2.{svg,png}`.

Alternatives considered and rejected: duplicate the loaders in the new script
(violates DRY); rewrite `plot.py` to emit the composite too (disturbs a
manuscript-critical working script mid-rebuttal).

## Tooling / styling

- ultraplot drives figure creation, rc, layout, `abc` lettering, auto-spacing.
- seaborn is only the per-axes drawing primitive — its axes-level functions
  accept `ax=`, and ultraplot axes are matplotlib `Axes` subclasses.
- Do **not** call `sns.set_theme` / `sns.set_context` in the composite, so
  seaborn's rc does not fight ultraplot's. Per-panel font sizing via
  `ax.format(...)` / `uplt.rc`.

## Prerequisite

`ultraplot` is **not installed** in any pixi env and is absent from `pixi.toml`.
Add it to the `default`/`bench` feature (conda-forge `ultraplot`, falling back to
PyPI) before implementation. This is step 0 of the plan.

## Outputs

- `figures/figure2.svg` — vector, the deliverable.
- `figures/figure2.png` — ~200 dpi preview.
- `figures/basenji2_rho.npz` — small ρ cache (new).

## Out of scope

- Panel E (GPU utilization screenshot) and Figure 1 schematic — remain
  Figma/Claude-Design composites.
- Supplementary figures (Supp. Fig. 1/2/3) — unchanged; still emitted by
  `plot.py` / `supplement.md`.
- The Google-Docs / citation-fixing downstream steps.
