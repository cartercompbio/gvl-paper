#! /usr/bin/env python
"""Finalize the memory track manually.

The orphaned nextflow driver completed every memory cell on disk but died
without committing its cache, so COMBINE_MEMORY/PLOT_MEMORY never ran. This
harvests the newest valid per-cell memory CSV from the work dirs (substituting
the freshly regenerated pgen q2048 cell), concatenates per method exactly like
combine_csvs.py, and writes the combined CSVs to results_recover/.
"""

import glob
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VALID_N = {10, 32, 100, 316, 1000, 3202}
METHODS = ["svar", "bcf", "pgen", "presubset_bcf"]

# regenerated cell that replaces the lost 2-row pgen q2048 partial
OVERRIDE = {
    ("pgen", 2048, 3202): HERE / "pgen_q2048_n3202_memory.csv",
}


def pick_cells():
    cells = defaultdict(list)  # (method,q,n) -> [(mtime,path)]
    for p in glob.glob(str(ROOT / "work/*/*/*_memory.csv")):
        b = os.path.basename(p)
        m = re.match(r"(.+)_q(\d+)_n(\d+)_memory.csv", b)
        if not m:
            continue
        meth, q, n = m.group(1), int(m.group(2)), int(m.group(3))
        if n not in VALID_N:
            continue
        cells[(meth, q, n)].append((os.path.getmtime(p), p))
    chosen = {}
    for key, v in cells.items():
        if key in OVERRIDE:
            chosen[key] = str(OVERRIDE[key])
        else:
            v.sort(reverse=True)
            chosen[key] = v[0][1]
    # apply overrides even if no work-dir cell existed
    for key, path in OVERRIDE.items():
        chosen[key] = str(path)
    return chosen


def main():
    chosen = pick_cells()
    ok = True
    for meth in METHODS:
        files = [p for (m, q, n), p in chosen.items() if m == meth]
        files.sort()
        frames = []
        for f in files:
            df = pl.read_csv(f)
            frames.append(df)
        combined = pl.concat(frames).sort(
            ["query_length", "replicate"], maintain_order=True
        )
        n_cells = len(files)
        n_rows = combined.height
        out = HERE / f"{meth}_memory.csv"
        combined.write_csv(out)
        bad = combined.filter(
            (pl.col("peak_rss_bytes").is_null()) | (pl.col("peak_rss_bytes") <= 0)
        ).height
        print(f"{meth}: {n_cells} cells -> {n_rows} rows ({bad} bad rss) -> {out.name}")
        if n_cells != 19:
            print(f"  WARNING: expected 19 cells, got {n_cells}")
            ok = False
    if not ok:
        sys.exit(1)

    # plots into results_recover/
    combined_csvs = [str(HERE / f"{m}_memory.csv") for m in METHODS]
    print("\nplotting...")
    subprocess.run(
        ["python", str(ROOT / "bin" / "plot_memory.py"), *combined_csvs,
         "--output-dir", str(HERE)],
        check=True,
    )
    print("done -> results_recover/{memory_plot,n_memory_plot}.{png,svg,pdf}")


if __name__ == "__main__":
    main()
