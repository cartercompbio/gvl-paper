# GVL 0.26.0 Throughput Parity Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure GVL 0.26.0 haplotype/track dataloading throughput (default + single-buffered) against the v0.6.1 manuscript baseline on an aligned, reduced sweep, to decide whether 0.26.0 recovers ~parity and fixes the 0.24.1 OOM.

**Architecture:** A standalone probe harness under `hap_track_throughput/bin_gvl026/` (mirroring the `bin_gvl061/` pattern, not wired into `benchmark.nf`). Pure helpers (grid math, MiB/s) are unit-tested; the dataloader timing loop is verified by smoke runs. A new `bench026` pixi env pins `genvarloader==0.26.0`. Datasets are reused when 0.26.0 can open them (TCGA), rebuilt when not (1KGP). A comparison script joins probe results to the baseline CSVs and emits a parity table + plot.

**Tech Stack:** Python 3.12, genvarloader 0.26.0 (CPU torch), numba, polars, cyclopts, pytest, seaborn/matplotlib, pixi.

---

## Reference facts (verified against the repo — do not re-derive)

- **Baseline CSVs** (already on disk, do NOT regenerate): `results/hap_results.csv` and `results/track_results.csv`. Schema: `dataset,threads,seqlen,batch_size,throughput (MiB/s)`. Dataset names: `1kgp`, `tcga-atac`, `ukbb`. Tracks baseline exists for `tcga-atac` only.
- **Reusable datasets (0.24.x-written):** `hap_track_throughput/data/datasets/tcga-atac/seqlen_{2048,16384,131072,1048576}.gvl` — 61 samples, `max_jitter=0`, contain both `genotypes/` and `intervals/read-depth/` (serve TCGA haps AND TCGA tracks). **1KGP is NOT built** — must be created (Task 6).
- **References:** 1KGP FASTA `/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa`; TCGA FASTA `/cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa`. 1KGP variants (PGEN) `/carter/users/dlaub/data/1kGP/plink2/hg38.norm.pgen` (+ `.pvar.zst`, `.psam`).
- **Probe grid (aligned to baseline cells, all confirmed present):** threads ∈ {1, 16, 64}; per seqlen, 3 batch sizes at fixed `npb = seqlen × batch_size` targets `2**21, 2**25, 2**29`:

  | seqlen | bs @npb 2²¹ | bs @npb 2²⁵ | bs @npb 2²⁹ |
  |--------|-------------|-------------|-------------|
  | 2048   | 1024        | 16384       | 262144      |
  | 16384  | 128         | 2048        | 32768       |
  | 131072 | 16          | 256         | 4096        |
  | 1048576| 2           | 32          | 512         |

  `n_batches = clip(2**29 // npb, 10, 200)` → 200 @2²¹, 16 @2²⁵, 10 @2²⁹.
- **Replicates:** 5 (matches `bin_gvl061`, the version that produced the baseline).
- **dl modes:** `none` (plain DataLoader) and `buffered` (single-buffered; `buffer_bytes=2*2**30` = 2 GiB).
- **MiB/s convention (matches `bin_gvl061`):** `total_output_bytes / seconds / 2**20`. Haps element size 1 (`S1`), tracks 4 (`float32`).
- **Output dir:** new top-level `results_gvl026/` (sibling of `results_gvl061/`).
- **All `pixi run` commands** must run from the repo root `/carter/users/dlaub/projects/gvl-paper`.

---

## File Structure

- Create `hap_track_throughput/bin_gvl026/_probe_common.py` — pure helpers: `batch_for_npb`, `n_batches_for`, `n_bytes`, `mib_per_s`, `PROBE_NPB_EXPS`.
- Create `hap_track_throughput/bin_gvl026/tests/test_probe_common.py` — unit tests for the pure helpers.
- Create `hap_track_throughput/bin_gvl026/make_probe_grid.py` — emit `threads,batch_size,n_batches` CSV for one seqlen.
- Create `hap_track_throughput/bin_gvl026/benchmark_dl.py` — the timed dataloader sweep (haps/tracks × none/buffered), emits MiB/s.
- Create `hap_track_throughput/bin_gvl026/probe026.sh` — idempotent driver over (dataset, mode, dl-mode) → `results_gvl026/`.
- Create `hap_track_throughput/bin_gvl026/compare_to_baseline.py` — join probe vs baseline, emit `results_gvl026/parity_summary.csv` + `figures/gvl026_parity.png`.
- Modify `pixi.toml` — add `bench026` feature + environment.
- Modify `CLAUDE.md` — add a `bench026` bullet to the version-sensitivity section.

---

## Task 1: Add the `bench026` pixi environment

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Add the `bench026` environment to `[environments]`**

In `pixi.toml`, after the `bench061 = ["bench061"]` line (currently `pixi.toml:17`), add:

```toml
# Pinned to genvarloader 0.26.0 — the release that directly targets the GVL >=0.21
# throughput/OOM regressions (see GenVarLoader/docs/superpowers/REGRESSIONS.md). Used ONLY by the
# parity probe in hap_track_throughput/bin_gvl026/ to compare against the 0.6.1 baseline
# (results/{hap,track}_results.csv). CPU torch — the benchmark has no GPU workload.
bench026 = ["bench026"]
```

- [ ] **Step 2: Add the `bench026` feature block**

In `pixi.toml`, after the `feature.bench061` block (ends at `pixi.toml:61`), add:

```toml
[feature.bench026.dependencies]
python = "3.12.*"

[feature.bench026.pypi-dependencies]
genvarloader = "==0.26.0"
torch = { version = "*", index = "https://download.pytorch.org/whl/cpu" }
cyclopts = ">=4.5.1, <5"
polars = ">=1"
psutil = "*"
pytest = "*"
```

- [ ] **Step 3: Resolve the environment**

Run: `cd /carter/users/dlaub/projects/gvl-paper && pixi install -e bench026`
Expected: solve + install completes. If the solver fails on `numpy`/`numba` pins from other features, add the minimal pin it reports under `[feature.bench026.pypi-dependencies]` (e.g. `numpy = "<2"`) and re-run. Do NOT touch other feature blocks.

- [ ] **Step 4: Verify the pinned version imports and reports 0.26.0**

Run: `pixi run -e bench026 python -c "import genvarloader as gvl; print(gvl.__version__)"`
Expected: prints `0.26.0` (or `0.26.x`). 

- [ ] **Step 5: Format-compatibility gate — can 0.26.0 open the existing TCGA dataset?**

Run:
```bash
pixi run -e bench026 python -c "
import genvarloader as gvl
ds = (gvl.Dataset.open(
        'hap_track_throughput/data/datasets/tcga-atac/seqlen_2048.gvl',
        '/cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa')
      .with_seqs('haplotypes').with_tracks(False).with_len(2048))
b = ds[0:2]
print('OK', type(b).__name__, getattr(b, 'shape', None))
"
```
Expected: prints `OK ...` with a 2-row batch. **If it raises a format/version error, the TCGA datasets must be rebuilt** — record this and add a TCGA build step modeled on Task 6 (using `--bigwig-table` for tracks); the rest of the plan is unchanged. If it succeeds, the TCGA datasets are reused as-is.

- [ ] **Step 6: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "env: add bench026 (genvarloader 0.26.0) for the parity probe"
```

---

## Task 2: Pure helpers + unit tests (`_probe_common.py`)

**Files:**
- Create: `hap_track_throughput/bin_gvl026/_probe_common.py`
- Test: `hap_track_throughput/bin_gvl026/tests/test_probe_common.py`

- [ ] **Step 1: Write the failing tests**

Create `hap_track_throughput/bin_gvl026/tests/test_probe_common.py`:

```python
import numpy as np
import pytest

from _probe_common import (
    PROBE_NPB_EXPS,
    batch_for_npb,
    n_batches_for,
    n_bytes,
    mib_per_s,
)


def test_probe_npb_exps_are_small_mid_large():
    assert PROBE_NPB_EXPS == (21, 25, 29)


@pytest.mark.parametrize(
    "seqlen,npb_exp,expected_bs",
    [
        (2048, 21, 1024), (2048, 25, 16384), (2048, 29, 262144),
        (16384, 21, 128), (16384, 25, 2048), (16384, 29, 32768),
        (131072, 21, 16), (131072, 25, 256), (131072, 29, 4096),
        (1048576, 21, 2), (1048576, 25, 32), (1048576, 29, 512),
    ],
)
def test_batch_for_npb_matches_baseline_cells(seqlen, npb_exp, expected_bs):
    assert batch_for_npb(seqlen, npb_exp) == expected_bs


def test_batch_for_npb_requires_power_of_two_seqlen_divisibility():
    # npb must be >= seqlen (batch_size >= 1)
    with pytest.raises(ValueError):
        batch_for_npb(2048, 5)  # 2**5 < 2048


def test_n_batches_for_clips_to_10_200():
    assert n_batches_for(2 ** 21) == 200   # 2**29 // 2**21 = 256 -> clip 200
    assert n_batches_for(2 ** 25) == 16     # 2**29 // 2**25 = 16
    assert n_batches_for(2 ** 29) == 10     # 2**29 // 2**29 = 1 -> clip 10


def test_n_bytes_numpy_and_object_with_numel():
    arr = np.zeros((4, 2048), dtype="S1")
    assert n_bytes(arr) == 4 * 2048 * 1
    farr = np.zeros((4, 2048), dtype=np.float32)
    assert n_bytes(farr) == 4 * 2048 * 4

    class FakeTensor:
        def numel(self):
            return 8
        def element_size(self):
            return 4
    assert n_bytes(FakeTensor()) == 32


def test_mib_per_s():
    # 2**20 bytes in 1 second == 1 MiB/s
    assert mib_per_s(2 ** 20, 1.0) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput/bin_gvl026 && pixi run -e bench026 python -m pytest tests/test_probe_common.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '_probe_common'`.

- [ ] **Step 3: Implement `_probe_common.py`**

Create `hap_track_throughput/bin_gvl026/_probe_common.py`:

```python
"""Pure helpers shared by the GVL 0.26.0 parity-probe scripts.

No genvarloader / torch imports here so the math stays unit-testable under any env.
"""

from __future__ import annotations

# Small / mid / large fetch sizes, as log2(nucleotides-per-batch). Each value is
# present in the v0.6.1 baseline grid for every probe seqlen, so the parity join matches.
PROBE_NPB_EXPS: tuple[int, ...] = (21, 25, 29)


def batch_for_npb(seqlen: int, npb_exp: int) -> int:
    """batch_size such that seqlen * batch_size == 2**npb_exp.

    Probe seqlens are powers of two, so this is exact. Raises if 2**npb_exp < seqlen
    (would imply batch_size < 1).
    """
    npb = 2 ** npb_exp
    if npb < seqlen:
        raise ValueError(f"npb 2**{npb_exp}={npb} < seqlen {seqlen}: batch_size would be < 1")
    bs, rem = divmod(npb, seqlen)
    if rem != 0:
        raise ValueError(f"seqlen {seqlen} does not divide npb 2**{npb_exp}")
    return bs


def n_batches_for(npb: int) -> int:
    """Number of measured batches for a cell: clip(2**29 // npb, 10, 200)."""
    raw = (2 ** 29) // max(1, npb)
    return min(200, max(10, raw))


def n_bytes(batch) -> int:
    """Total bytes in a batch, supporting numpy arrays and torch-like tensors."""
    if hasattr(batch, "itemsize") and hasattr(batch, "size"):  # numpy ndarray
        return int(batch.size) * int(batch.itemsize)
    return int(batch.numel()) * int(batch.element_size())  # torch.Tensor


def mib_per_s(total_bytes: int, seconds: float) -> float:
    """Throughput in MiB/s. Matches bin_gvl061's convention."""
    return total_bytes / seconds / 2 ** 20
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput/bin_gvl026 && pixi run -e bench026 python -m pytest tests/test_probe_common.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hap_track_throughput/bin_gvl026/_probe_common.py hap_track_throughput/bin_gvl026/tests/test_probe_common.py
git commit -m "feat(probe026): pure grid/throughput helpers + unit tests"
```

---

## Task 3: Probe grid generator (`make_probe_grid.py`)

**Files:**
- Create: `hap_track_throughput/bin_gvl026/make_probe_grid.py`

- [ ] **Step 1: Implement the generator**

Create `hap_track_throughput/bin_gvl026/make_probe_grid.py`:

```python
#! /usr/bin/env python
"""Emit the reduced parity-probe grid (threads x batch_size x n_batches) for one seqlen.

Columns match make_launch_grid.py so benchmark_dl.py can read either:
    threads,batch_size,n_batches
"""

from pathlib import Path

import cyclopts

from _probe_common import PROBE_NPB_EXPS, batch_for_npb, n_batches_for

THREADS = (1, 16, 64)


def main(length: int, output: Path | None = None):
    import polars as pl

    rows = []
    for npb_exp in PROBE_NPB_EXPS:
        bs = batch_for_npb(length, npb_exp)
        nb = n_batches_for(2 ** npb_exp)
        for t in THREADS:
            rows.append({"threads": t, "batch_size": bs, "n_batches": nb})

    if output is None:
        output = Path.cwd() / f"probe_grid_{length}.csv"
    pl.from_dicts(rows).write_csv(output)
    print(f"WROTE {output} ({len(rows)} rows)")


if __name__ == "__main__":
    cyclopts.run(main)
```

- [ ] **Step 2: Verify it produces 9 rows with the expected cells**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput/bin_gvl026
pixi run -e bench026 python make_probe_grid.py 16384 --output /tmp/g16384.csv && cat /tmp/g16384.csv
```
Expected: header + 9 rows; batch_size values {128, 2048, 32768} each paired with threads {1,16,64}; n_batches {200,16,10} respectively.

- [ ] **Step 3: Commit**

```bash
git add hap_track_throughput/bin_gvl026/make_probe_grid.py
git commit -m "feat(probe026): reduced parity grid generator"
```

---

## Task 4: The timed dataloader sweep (`benchmark_dl.py`)

**Files:**
- Create: `hap_track_throughput/bin_gvl026/benchmark_dl.py`

- [ ] **Step 1: Implement the sweep**

Create `hap_track_throughput/bin_gvl026/benchmark_dl.py`:

```python
#! /usr/bin/env python
"""Timed GVL 0.26.0 dataloader throughput sweep for the parity probe.

Sweeps a (threads, batch_size, n_batches) grid for one dataset + seqlen, in one
output mode (haps|tracks) and one dataloader mode (none|buffered), and records
throughput in MiB/s using the same convention as bin_gvl061 (directly comparable
to results/{hap,track}_results.csv).

Output CSV schema:
    dataset,backend,mode,dl_mode,threads,seqlen,batch_size,throughput (MiB/s)
"""

from pathlib import Path

import cyclopts

from _probe_common import mib_per_s, n_bytes


def bench(
    results: Path,
    ds_path: Path,
    length: int,
    fasta: Path,
    grid_file: Path,
    mode: str = "haps",          # "haps" or "tracks"
    dl_mode: str = "none",       # "none" or "buffered"
    dataset: str = "",
    backend: str = "gvl026",
    buffer_bytes: int = 2 * 2 ** 30,
    burn_in: int = 1,
    replicates: int = 5,
):
    import os
    from time import perf_counter

    import genvarloader as gvl
    import numba as nb
    import polars as pl

    if mode == "haps":
        ds = (
            gvl.Dataset.open(ds_path, fasta)
            .with_seqs("haplotypes")
            .with_tracks(False)
            .with_len(length)
            .with_settings(deterministic=True)  # required by buffered haps
        )
    elif mode == "tracks":
        ds = (
            gvl.Dataset.open(ds_path, fasta)
            .with_seqs(None)
            .with_tracks("read-depth")
            .with_len(length)
        )
    else:
        raise ValueError(f"mode must be 'haps' or 'tracks', got {mode!r}")

    dataset = dataset or ds_path.parent.name
    max_threads = len(os.sched_getaffinity(0))
    grid = pl.read_csv(grid_file)
    assert int(grid["threads"].max()) <= max_threads  # type: ignore

    dl_kwargs = {} if dl_mode == "none" else {"mode": "buffered", "buffer_bytes": buffer_bytes}

    with open(results, "w") as f:
        f.write("dataset,backend,mode,dl_mode,threads,seqlen,batch_size,throughput (MiB/s)\n")
        f.flush()
        for n_thread, batch_size, n_batches in grid.iter_rows():
            nb.set_num_threads(n_thread)
            try:
                dl = ds.to_dataloader(batch_size=batch_size, shuffle=False, **dl_kwargs)
            except ValueError as e:
                # buffered: a single mini-batch can exceed buffer_bytes -> construction raises.
                # Record the cell as NaN so the sweep continues and the gap is visible.
                print(f"SKIP cell t={n_thread} bs={batch_size} ({dl_mode}): {e}", flush=True)
                for _ in range(replicates):
                    f.write(f"{dataset},{backend},{mode},{dl_mode},{n_thread},{length},{batch_size},nan\n")
                f.flush()
                continue

            for _ in range(replicates):
                n_yielded = 0
                total_bytes = 0
                t0 = perf_counter()
                done = False
                while not done:
                    for batch in dl:
                        if n_yielded == burn_in:
                            t0 = perf_counter()
                        if n_yielded >= burn_in:
                            total_bytes += n_bytes(batch)
                        n_yielded += 1
                        if n_yielded >= n_batches + burn_in:
                            done = True
                            break
                seconds = perf_counter() - t0
                tput = mib_per_s(total_bytes, seconds) if seconds > 0 else float("nan")
                f.write(f"{dataset},{backend},{mode},{dl_mode},{n_thread},{length},{batch_size},{tput}\n")
                f.flush()
            del dl


if __name__ == "__main__":
    cyclopts.run(bench)
```

Notes for the implementer:
- `with_settings(deterministic=True)` is intentional for haps — buffered haps raises without it; harmless for `none`.
- The `ValueError` catch handles the documented buffered precondition "a single mini-batch whose footprint exceeds the per-slot capacity raises" (e.g. tracks @ npb 2²⁹ = 2 GiB float32 vs the 2 GiB buffer). Per-row `flush()` preserves partial CSVs if a cell OOM-kills the process.
- `n_bytes(batch)` assumes one array per batch — guaranteed because haps disables tracks and tracks disables seqs.

- [ ] **Step 2: Commit (smoke-tested in Task 5)**

```bash
git add hap_track_throughput/bin_gvl026/benchmark_dl.py
git commit -m "feat(probe026): timed dataloader sweep (haps/tracks x none/buffered)"
```

---

## Task 5: Smoke-test `benchmark_dl.py` on the reused TCGA dataset

**Files:** (no new files — verification only)

- [ ] **Step 1: Build a tiny 1-cell grid**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput/bin_gvl026
printf "threads,batch_size,n_batches\n8,8,3\n" > /tmp/smoke_grid.csv
```

- [ ] **Step 2: Smoke-run haps, dl_mode=none**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput/bin_gvl026
pixi run -e bench026 python benchmark_dl.py /tmp/haps_none.csv \
  ../data/datasets/tcga-atac/seqlen_2048.gvl 2048 \
  /cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa \
  /tmp/smoke_grid.csv --mode haps --dl-mode none --dataset tcga-atac
cat /tmp/haps_none.csv
```
Expected: header + 5 rows (replicates), `dl_mode=none`, `throughput (MiB/s)` finite and > 0.

- [ ] **Step 3: Smoke-run haps, dl_mode=buffered**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput/bin_gvl026
pixi run -e bench026 python benchmark_dl.py /tmp/haps_buf.csv \
  ../data/datasets/tcga-atac/seqlen_2048.gvl 2048 \
  /cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa \
  /tmp/smoke_grid.csv --mode haps --dl-mode buffered --dataset tcga-atac
cat /tmp/haps_buf.csv
```
Expected: header + 5 rows, `dl_mode=buffered`, throughput finite > 0. If it raises about `num_workers` or `deterministic`, the dataloader config is wrong — fix `benchmark_dl.py` (do not set `num_workers`; ensure `with_settings(deterministic=True)`) and re-run.

- [ ] **Step 4: Smoke-run tracks, both dl-modes**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput/bin_gvl026
pixi run -e bench026 python benchmark_dl.py /tmp/trk_none.csv \
  ../data/datasets/tcga-atac/seqlen_2048.gvl 2048 \
  /cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa \
  /tmp/smoke_grid.csv --mode tracks --dl-mode none --dataset tcga-atac
pixi run -e bench026 python benchmark_dl.py /tmp/trk_buf.csv \
  ../data/datasets/tcga-atac/seqlen_2048.gvl 2048 \
  /cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa \
  /tmp/smoke_grid.csv --mode tracks --dl-mode buffered --dataset tcga-atac
cat /tmp/trk_none.csv /tmp/trk_buf.csv
```
Expected: both produce 5 finite, >0 throughput rows (at bs=8, npb=16384, well under the 2 GiB buffer).

- [ ] **Step 5: No code change needed → no commit.** If Steps 2–4 forced a fix to `benchmark_dl.py`, amend Task 4's commit:

```bash
git add hap_track_throughput/bin_gvl026/benchmark_dl.py
git commit --amend --no-edit
```

---

## Task 6: Build 1KGP probe datasets with 0.26.0

**Files:** (no new source files — reuses `bin/make_bed.py` and `bin/benchmark_write.py`)

> 1KGP is not built locally. Build 4 seqlen datasets with 0.26.0 itself (guarantees format compatibility). Uses the existing `make_bed.py` (100 regions/chrom, canonical) and `benchmark_write.py` (new-API `gvl.write`). All 3202 samples (no `samples=` subset) — throughput in MiB/s is per-batch and comparable to the baseline regardless of pool size.

- [ ] **Step 1: Make the probe BED (shared across seqlens is NOT valid — bed is per-length tiling; make one per seqlen in the loop below)**

(Done inside Step 2's loop.)

- [ ] **Step 2: Build all four 1KGP datasets**

Run (this is the heavy step; ~tens of minutes to a few hours total depending on seqlen):
```bash
cd /carter/users/dlaub/projects/gvl-paper
FASTA=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa
PGEN=/carter/users/dlaub/data/1kGP/plink2/hg38.norm.pgen
OUTDIR=hap_track_throughput/data/datasets_gvl026/1kgp
mkdir -p "$OUTDIR"
for L in 2048 16384 131072 1048576; do
  DS="$OUTDIR/seqlen_${L}.gvl"
  if [ -d "$DS" ]; then echo "SKIP $DS (exists)"; continue; fi
  pixi run -e bench026 python hap_track_throughput/bin/make_bed.py \
    "$L" "$FASTA" "/tmp/1kgp_${L}.bed" --canonical --n-samples 100
  pixi run -e bench026 python hap_track_throughput/bin/benchmark_write.py \
    "$OUTDIR/write_${L}.csv" "$PGEN" "/tmp/1kgp_${L}.bed" "$FASTA" \
    "$DS" "$L" native --dataset 1kgp --max-mem 16g --max-jitter 0
  echo "BUILT $DS"
done
```
Expected: four `seqlen_*.gvl` directories created under `hap_track_throughput/data/datasets_gvl026/1kgp/`, each containing `genotypes/` + `metadata.json`. `benchmark_write.py` runs under `bench026` because it uses the new `gvl.write` API (compatible with 0.26.0). `bin/` is not on PATH here, but `make_bed.py`/`benchmark_write.py` import only stdlib + installed packages, so invoking by path works; `benchmark_write.py`'s `from _mem_sampler import ...` is inside the `--measure-memory` branch only, so it is not triggered.

- [ ] **Step 3: Verify 0.26.0 opens a built 1KGP dataset as haps**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper
pixi run -e bench026 python -c "
import genvarloader as gvl
ds = (gvl.Dataset.open('hap_track_throughput/data/datasets_gvl026/1kgp/seqlen_2048.gvl',
        '/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa')
      .with_seqs('haplotypes').with_tracks(False).with_len(2048))
print('OK 1kgp', ds[0:2].shape)
"
```
Expected: `OK 1kgp (2, 2048)` (or `(2, 2, 2048)` if ploidy axis is present — either is fine; just no error).

- [ ] **Step 4: Commit the build artifacts metadata (datasets themselves are under gitignored `data/`)**

The datasets live under `hap_track_throughput/data/...` which is gitignored, so nothing to commit here. Record completion in the run log only. No commit.

---

## Task 7: Run the full probe

**Files:**
- Create: `hap_track_throughput/bin_gvl026/probe026.sh`

- [ ] **Step 1: Write the driver**

Create `hap_track_throughput/bin_gvl026/probe026.sh`:

```bash
#!/bin/bash
# Idempotent driver for the GVL 0.26.0 parity probe.
# Runs (dataset,output-mode) x dl-mode x seqlen over the reduced grid, writing
# one CSV per combo into results_gvl026/. Skips combos whose CSV already exists.
set -euo pipefail

ROOT=/carter/users/dlaub/projects/gvl-paper
BIN="$ROOT/hap_track_throughput/bin_gvl026"
OUT="$ROOT/results_gvl026"
mkdir -p "$OUT"

TCGA_FASTA=/cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa
KGP_FASTA=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa
TCGA_DS_DIR="$ROOT/hap_track_throughput/data/datasets/tcga-atac"
KGP_DS_DIR="$ROOT/hap_track_throughput/data/datasets_gvl026/1kgp"

SEQLENS=(2048 16384 131072 1048576)
DLMODES=(none buffered)

# combo := "dataset:output_mode:ds_dir:fasta"
COMBOS=(
  "1kgp:haps:$KGP_DS_DIR:$KGP_FASTA"
  "tcga-atac:haps:$TCGA_DS_DIR:$TCGA_FASTA"
  "tcga-atac:tracks:$TCGA_DS_DIR:$TCGA_FASTA"
)

cd "$BIN"
for combo in "${COMBOS[@]}"; do
  IFS=: read -r dataset omode ds_dir fasta <<< "$combo"
  for L in "${SEQLENS[@]}"; do
    grid="/tmp/probe_grid_${L}.csv"
    [ -f "$grid" ] || pixi run -e bench026 python make_probe_grid.py "$L" --output "$grid"
    ds="$ds_dir/seqlen_${L}.gvl"
    if [ ! -d "$ds" ]; then echo "MISSING dataset $ds — skipping"; continue; fi
    for dl in "${DLMODES[@]}"; do
      res="$OUT/${dataset}_${omode}_${L}_${dl}.csv"
      if [ -f "$res" ]; then echo "SKIP $res (exists)"; continue; fi
      echo "RUN $dataset $omode seqlen=$L dl=$dl"
      pixi run -e bench026 python benchmark_dl.py "$res" "$ds" "$L" "$fasta" "$grid" \
        --mode "$omode" --dl-mode "$dl" --dataset "$dataset"
    done
  done
done
echo "PROBE COMPLETE -> $OUT"
```

- [ ] **Step 2: Make it executable and dry-check the combo expansion**

Run: `chmod +x hap_track_throughput/bin_gvl026/probe026.sh && bash -n hap_track_throughput/bin_gvl026/probe026.sh && echo "syntax ok"`
Expected: `syntax ok`.

- [ ] **Step 3: Run the probe**

Run (long-running; consider `nohup`/SLURM — it needs up to 64 cores; on this repo the haps/tracks benches ran on `carter-cn-04`):
```bash
cd /carter/users/dlaub/projects/gvl-paper
nohup bash hap_track_throughput/bin_gvl026/probe026.sh > results_gvl026/probe.log 2>&1 &
```
Then monitor: `tail -f results_gvl026/probe.log`
Expected: produces `results_gvl026/{1kgp_haps,tcga-atac_haps,tcga-atac_tracks}_{2048,16384,131072,1048576}_{none,buffered}.csv` (24 files). Buffered tracks @ npb 2²⁹ may show `nan` rows (batch footprint > 2 GiB) — that is expected and handled, not a failure.

- [ ] **Step 4: Verify completeness**

Run: `ls results_gvl026/*.csv | wc -l && grep -L "throughput" results_gvl026/*.csv || echo "all have headers"`
Expected: 24 CSVs, all with the header line. Re-running `probe026.sh` fills any gaps (idempotent).

- [ ] **Step 5: Commit the driver + results**

```bash
git add hap_track_throughput/bin_gvl026/probe026.sh results_gvl026/*.csv results_gvl026/probe.log
git commit -m "feat(probe026): driver + 0.26.0 parity-probe throughput results"
```
(`results_gvl026/` is not gitignored — confirm with `git status` before committing; the baseline `results_gvl061/` precedent is to commit the CSVs.)

---

## Task 8: Compare to baseline + parity plot

**Files:**
- Create: `hap_track_throughput/bin_gvl026/compare_to_baseline.py`

- [ ] **Step 1: Implement the comparison**

Create `hap_track_throughput/bin_gvl026/compare_to_baseline.py`:

```python
#! /usr/bin/env python
"""Join the 0.26.0 probe results to the v0.6.1 baseline and report parity.

Outputs:
  results_gvl026/parity_summary.csv  — per (dataset,output_mode,dl_mode,threads,seqlen,batch_size):
       baseline MiB/s, 0.26.0 MiB/s (median over replicates), ratio = v026 / v061
  figures/gvl026_parity.png          — scatter, x=baseline, y=0.26.0, hue=dl_mode, facet=seqlen
"""

from pathlib import Path

import cyclopts


def main(
    results_dir: Path = Path("results_gvl026"),
    hap_baseline: Path = Path("results/hap_results.csv"),
    track_baseline: Path = Path("results/track_results.csv"),
    fig_out: Path = Path("figures/gvl026_parity.png"),
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    # --- load probe results ---
    probe = pl.concat(
        [pl.read_csv(p) for p in sorted(results_dir.glob("*_*_*_*.csv"))],
        how="vertical_relaxed",
    ).rename({"throughput (MiB/s)": "v026"})
    probe = (
        probe.drop_nulls("v026")
        .group_by(["dataset", "mode", "dl_mode", "threads", "seqlen", "batch_size"])
        .agg(pl.col("v026").median())
    )

    # --- load baselines, tag output mode ---
    hap = pl.read_csv(hap_baseline).with_columns(mode=pl.lit("haps"))
    trk = pl.read_csv(track_baseline).with_columns(mode=pl.lit("tracks"))
    base = (
        pl.concat([hap, trk], how="vertical_relaxed")
        .rename({"throughput (MiB/s)": "v061"})
        .group_by(["dataset", "mode", "threads", "seqlen", "batch_size"])
        .agg(pl.col("v061").median())
    )

    joined = probe.join(
        base, on=["dataset", "mode", "threads", "seqlen", "batch_size"], how="left"
    ).with_columns(ratio=(pl.col("v026") / pl.col("v061")))

    results_dir.mkdir(exist_ok=True)
    summary = results_dir / "parity_summary.csv"
    joined.sort(["dataset", "mode", "dl_mode", "seqlen", "batch_size", "threads"]).write_csv(summary)
    print(f"WROTE {summary} ({joined.height} rows)")

    matched = joined.drop_nulls("v061")
    print("Median 0.26.0/0.6.1 ratio by dl_mode:")
    print(matched.group_by("dl_mode").agg(pl.col("ratio").median()).sort("dl_mode"))

    # --- parity scatter ---
    pdf = matched.to_pandas()
    fig_out.parent.mkdir(exist_ok=True)
    g = sns.relplot(
        data=pdf, x="v061", y="v026", hue="dl_mode", style="mode",
        col="seqlen", col_wrap=2, facet_kws={"sharex": False, "sharey": False},
    )
    for ax in g.axes.flat:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], ls="--", c="grey", lw=1)  # parity line
        ax.set_xscale("log"); ax.set_yscale("log")
    g.set_axis_labels("v0.6.1 MiB/s", "v0.26.0 MiB/s")
    g.savefig(fig_out, dpi=150, bbox_inches="tight")
    print(f"WROTE {fig_out}")


if __name__ == "__main__":
    cyclopts.run(main)
```

- [ ] **Step 2: Run it (under the default `bench` env — has seaborn/matplotlib)**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper
pixi run python hap_track_throughput/bin_gvl026/compare_to_baseline.py
```
Expected: prints the per-`dl_mode` median ratio table, writes `results_gvl026/parity_summary.csv` and `figures/gvl026_parity.png`. Points on/above the dashed parity line = 0.26.0 met/beat 0.6.1.

- [ ] **Step 3: Sanity-check the join matched cells**

Run: `pixi run python -c "import polars as pl; d=pl.read_csv('results_gvl026/parity_summary.csv'); print('rows', d.height, 'matched', d.drop_nulls('v061').height)"`
Expected: `matched` > 0 and close to `rows` (a few buffered-`nan` tracks cells may be dropped). If `matched == 0`, the join keys are misaligned — check that probe `dataset` strings are exactly `1kgp`/`tcga-atac` and `mode` is `haps`/`tracks`.

- [ ] **Step 4: Commit**

```bash
git add hap_track_throughput/bin_gvl026/compare_to_baseline.py results_gvl026/parity_summary.csv figures/gvl026_parity.png
git commit -m "feat(probe026): baseline comparison + parity plot"
```

---

## Task 9: Document the env in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the `bench026` bullet**

In `CLAUDE.md`, in the "GenVarLoader version sensitivity" section, after the `feature.bench` bullet (the `genvarloader >=0.24.1` one), add:

```markdown
- `feature.bench026` → `genvarloader ==0.26.0`: the release targeting the >=0.21 throughput/OOM
  regressions. Used **only** by the parity probe in `hap_track_throughput/bin_gvl026/`, which
  compares 0.26.0 (default + single-`buffered` dataloading) against the 0.6.1 baseline
  (`results/{hap,track}_results.csv`); probe outputs land in `results_gvl026/`. CPU torch (no GPU
  workload). Do **not** repoint the manuscript baseline CSVs to this env.
```

- [ ] **Step 2: Verify the section reads coherently**

Run: `pixi run python -c "print(open('CLAUDE.md').read().count('bench026'))"`
Expected: `>= 1`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document bench026 parity-probe env in CLAUDE.md"
```

---

## Final verification

- [ ] All unit tests pass: `cd hap_track_throughput/bin_gvl026 && pixi run -e bench026 python -m pytest tests/ -v`
- [ ] `results_gvl026/` has 24 throughput CSVs + `parity_summary.csv`.
- [ ] `figures/gvl026_parity.png` exists and shows points relative to the parity line.
- [ ] The per-`dl_mode` median ratio (from Task 8 Step 2) is reported — this is the headline answer: does 0.26.0 (esp. buffered) reach ~parity with 0.6.1?

---

## Self-review notes (spec coverage)

- Goal/success criterion → Final verification + Task 8.
- `bench026` env (new, CPU torch) → Task 1; CLAUDE.md → Task 9.
- Reuse-first datasets (TCGA reused via open-gate Task 1 Step 5; 1KGP rebuilt) → Tasks 1, 6. **Spec said reuse 1KGP; reality: only TCGA is built, so 1KGP is a rebuild — covered in Task 6.**
- Reduced grid aligned to baseline cells → Task 3 + Reference table (all cells verified present in baseline).
- Both dl-modes (none + buffered, 2 GiB) → Task 4, driver Task 7.
- MiB/s emitted directly, comparable schema → Task 2 (`mib_per_s`) + Task 4.
- Standalone driver (not benchmark.nf) → Task 7.
- Analysis/parity artifact → Task 8.
- Out of scope (full sweep, UKBB, memory, double_buffered) → not included, as specified.
