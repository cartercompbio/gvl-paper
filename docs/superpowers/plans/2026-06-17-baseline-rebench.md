# FASTA / pyBigWig Baseline Re-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-measure the FASTA (haplotype) and pyBigWig (track) baselines apples-to-apples with the GVL 0.27 grid — same `measure_cell` timing loop, schema, region tiles, (threads × batch × seqlen) grid, and node (cn-03) — so the manuscript's re-based speedups are honest.

**Architecture:** A standalone `benchmark_baseline.py` reuses the existing pysam `Ref` and `gvl.BigWigs` dataset classes but drives them through `_bench_common.measure_cell` (the exact loop the 0.27 GVL numbers use). Thread counts are swept by `taskset`-limited CPU sets (matching how the original baseline limited cores), `num_workers = n_cpus - 1` (answers reviewer R2-min1). A `run_baselines.sbatch` runs FASTA then pyBigWig sequentially on cn-03; `compute_speedups.py` joins the results to the GVL CSVs.

**Tech Stack:** Python 3.12, cyclopts, polars, pysam, genvarloader (`gvl.BigWigs`), torch `DataLoader`, pytest; SLURM; pixi env `bench027`.

**Spec:** `docs/superpowers/specs/2026-06-17-baseline-rebench-design.md`. Branch: `spec/baseline-rebench`.

---

## File structure

- Create `hap_track_throughput/bin/benchmark_baseline.py` — the benchmark script: pure helpers (`batch_bytes`, `cell_fits`, `select_cells`), the `Ref`/`BigWigDataset` classes, and the `measure_cell` driver. One responsibility: measure baseline throughput for one `--kind` at one thread count.
- Create `hap_track_throughput/bin/tests/test_benchmark_baseline.py` — unit tests for the pure helpers + a smoke test of the FASTA/pyBigWig drivers over a tiny synthetic FASTA/BigWig.
- Create `hap_track_throughput/run_baselines.sbatch` — cn-03 driver: loops kind × thread-count via `taskset`, sequential.
- Create `hap_track_throughput/bin_gvl027/compute_speedups.py` — join baselines to `results_gvl027/{haps,tracks}/*_none.csv`, emit `speedup = GVL_max / baseline_max` per (mode, dataset, seqlen) + absolute MiB/s for the A100 check.
- Reuse (no edit): `bin/_bench_common.py` (`measure_cell`, `THROUGHPUT_HEADER`, `n_bytes`, `mib_per_s`), `bin/make_launch_grid.py`, `bin/make_bed.py`.

---

## Task 1: Verify the `bench027` env has pysam + gvl.BigWigs

**Files:** none (env check); may modify `pixi.toml`.

- [ ] **Step 1: Check imports in bench027**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper
pixi run -e bench027 python -c "import pysam, genvarloader as gvl; gvl.BigWigs; print('ok')"
```
Expected: `ok`. If `ModuleNotFoundError: pysam`, do Step 2; else skip to Task 2.

- [ ] **Step 2: Add pysam to bench027 (only if missing)**

Run:
```bash
pixi add --feature bench027 "pysam!=0.23.1"
pixi run -e bench027 python -c "import pysam; print(pysam.__version__)"
```
Expected: a version prints.

- [ ] **Step 3: Commit (only if pixi.toml changed)**

```bash
git add pixi.toml pixi.lock
git commit -m "chore(bench027): add pysam for the baseline re-bench"
```

---

## Task 2: Pure helpers + unit tests (memory guard, byte accounting, cell selection)

**Files:**
- Create: `hap_track_throughput/bin/benchmark_baseline.py` (helpers only for now)
- Test: `hap_track_throughput/bin/tests/test_benchmark_baseline.py`

- [ ] **Step 1: Write the failing tests**

```python
# hap_track_throughput/bin/tests/test_benchmark_baseline.py
import sys
from pathlib import Path

import polars as pl
import pytest

BIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BIN))

from benchmark_baseline import batch_bytes, cell_fits, select_cells  # noqa: E402


def test_batch_bytes_fasta_is_one_byte_per_bp():
    # 4 instances x 2048 bp x 1 byte
    assert batch_bytes(batch_size=4, seqlen=2048, bytes_per_bp=1) == 4 * 2048


def test_batch_bytes_bigwig_is_four_bytes_per_bp():
    assert batch_bytes(batch_size=4, seqlen=2048, bytes_per_bp=4) == 4 * 2048 * 4


def test_cell_fits_rejects_oversized_prefetch():
    # batch 8192 x 1Mbp x 1 byte = 8 GiB; x (workers+1) x prefetch blows past 96 GiB
    assert not cell_fits(
        batch_size=8192, seqlen=1_048_576, bytes_per_bp=1,
        num_workers=31, prefetch_factor=2, mem_cap_bytes=96 * 2**30,
    )


def test_cell_fits_accepts_small_cell():
    assert cell_fits(
        batch_size=32, seqlen=2048, bytes_per_bp=1,
        num_workers=31, prefetch_factor=2, mem_cap_bytes=96 * 2**30,
    )


def test_select_cells_filters_threads_and_dedups_batches(tmp_path):
    grid = pl.DataFrame(
        {
            "threads": [1, 4, 1, 4],
            "batch_size": [2, 2, 32, 32],
            "n_batches": [256, 256, 16, 16],
        }
    )
    p = tmp_path / "grid_2048.csv"
    grid.write_csv(p)
    cells = select_cells(p, threads=4)
    # only threads==4 rows, as (batch_size, n_batches) tuples
    assert cells == [(2, 256), (32, 16)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e bench027 pytest hap_track_throughput/bin/tests/test_benchmark_baseline.py -q`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'batch_bytes'`.

- [ ] **Step 3: Implement the helpers**

Create `hap_track_throughput/bin/benchmark_baseline.py`:
```python
#! /usr/bin/env python
"""Apples-to-apples FASTA / pyBigWig baselines for the GVL 0.27 throughput grid.

Reuses the pysam `Ref` and `gvl.BigWigs` dataset readers but drives them through
`_bench_common.measure_cell` so throughput (MiB/s) is directly comparable to the
GVL numbers in results_gvl027/{haps,tracks}/*_none.csv. Thread counts are swept
by the caller via taskset (one invocation per thread count); num_workers is the
allocated CPU count minus one (the main process), matching a fully-optimized
PyTorch multiprocessing DataLoader (reviewer R2-min1).
"""

from pathlib import Path

from cyclopts import run


def batch_bytes(*, batch_size: int, seqlen: int, bytes_per_bp: int) -> int:
    return batch_size * seqlen * bytes_per_bp


def cell_fits(
    *,
    batch_size: int,
    seqlen: int,
    bytes_per_bp: int,
    num_workers: int,
    prefetch_factor: int,
    mem_cap_bytes: int,
) -> bool:
    """A cell fits if its prefetched batches stay under the RAM cap.

    torch's DataLoader prefetches prefetch_factor batches per worker; the main
    process also holds one. Approximate peak as (num_workers * prefetch_factor + 1)
    decoded batches resident at once.
    """
    resident = (num_workers * prefetch_factor + 1) * batch_bytes(
        batch_size=batch_size, seqlen=seqlen, bytes_per_bp=bytes_per_bp
    )
    return resident <= mem_cap_bytes


def select_cells(grid_file: Path, *, threads: int) -> list[tuple[int, int]]:
    """Distinct (batch_size, n_batches) cells for the given thread count."""
    import polars as pl

    grid = pl.read_csv(grid_file)
    rows = (
        grid.filter(pl.col("threads") == threads)
        .select("batch_size", "n_batches")
        .unique()
        .sort("batch_size")
    )
    return [(int(b), int(n)) for b, n in rows.iter_rows()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e bench027 pytest hap_track_throughput/bin/tests/test_benchmark_baseline.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add hap_track_throughput/bin/benchmark_baseline.py hap_track_throughput/bin/tests/test_benchmark_baseline.py
git commit -m "feat(baseline): pure helpers (batch_bytes, cell_fits, select_cells) + tests"
```

---

## Task 3: FASTA driver (`--kind fasta`) + smoke test

**Files:**
- Modify: `hap_track_throughput/bin/benchmark_baseline.py`
- Test: `hap_track_throughput/bin/tests/test_benchmark_baseline.py`

- [ ] **Step 1: Write the failing smoke test**

Append to the test file:
```python
def _write_tiny_fasta(tmp_path) -> Path:
    import subprocess
    fa = tmp_path / "tiny.fa"
    # one contig "chr1", 10 kb of ACGT so 2048-bp reads have room
    seq = ("ACGT" * 2560)[:10_000]
    fa.write_text(">chr1\n" + "\n".join(seq[i:i+60] for i in range(0, len(seq), 60)) + "\n")
    subprocess.run(["samtools", "faidx", str(fa)], check=True)
    return fa


def _write_bed(tmp_path, length, n_regions) -> Path:
    bed = tmp_path / f"tile_{length}.bed"
    lines = [f"chr1\t{i*length}\t{(i+1)*length}" for i in range(n_regions)]
    bed.write_text("\n".join(lines) + "\n")
    return bed


def test_fasta_driver_writes_comparable_schema(tmp_path):
    import polars as pl
    from benchmark_baseline import run_kind

    fa = _write_tiny_fasta(tmp_path)
    beds = {2048: _write_bed(tmp_path, 2048, 4)}
    grid = tmp_path / "grid_2048.csv"
    pl.DataFrame({"threads": [1], "batch_size": [2], "n_batches": [3]}).write_csv(grid)
    out = tmp_path / "fasta.csv"

    run_kind(
        kind="fasta", threads=1, fasta=fa, bigwig_table=None,
        bed_dir=tmp_path, grid_dir=tmp_path, seqlens=[2048],
        results=out, n_samples=4, mem_cap_bytes=96 * 2**30,
        burn_in=1, replicates=1, time_limit_s=5.0, min_batches=1,
    )

    df = pl.read_csv(out)
    assert df.columns == [
        "dataset", "backend", "dl_mode", "threads", "seqlen", "batch_size",
        "n_batches_measured", "total_bytes", "duration_ns", "throughput (MiB/s)",
    ]
    assert (df["backend"] == "fasta").all()
    assert (df["dl_mode"] == "none").all()
    assert df["throughput (MiB/s)"].drop_nulls().gt(0).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e bench027 pytest hap_track_throughput/bin/tests/test_benchmark_baseline.py::test_fasta_driver_writes_comparable_schema -q`
Expected: FAIL — `ImportError: cannot import name 'run_kind'`.

- [ ] **Step 3: Implement the `Ref` dataset, the driver, and the `run_kind` orchestration**

Append to `benchmark_baseline.py` (the `Ref` class is copied verbatim from `benchmark_ref.py`; `BigWigDataset` is added in Task 4):
```python
def _make_ref_dataset(fasta: Path, bed: Path, n_samples: int):
    import numpy as np
    import polars as pl
    import pysam
    from torch.utils.data import Dataset

    class Ref(Dataset):
        def __init__(self, path, bed, n_samples):
            self.path = path
            self.fasta = None
            self.bed = pl.read_csv(
                bed, separator="\t", has_header=False,
                new_columns=["contig", "start", "end"],
                schema_overrides={"contig": pl.Utf8},
            )
            self.n_samples = n_samples
            bed_ucsc = self.bed["contig"].str.contains("chr").any()
            with pysam.FastaFile(str(self.path)) as f:
                fa_ucsc = any(c.startswith("chr") for c in f.references)
            if not bed_ucsc and fa_ucsc:
                self.bed = self.bed.with_columns("chr" + pl.col("contig"))
            elif bed_ucsc and not fa_ucsc:
                self.bed = self.bed.with_columns(pl.col("contig").str.slice(3))

        @property
        def shape(self):
            return (self.bed.height, self.n_samples)

        def __len__(self):
            return self.bed.height * self.n_samples

        def __getitem__(self, index):
            if self.fasta is None:
                self.fasta = pysam.FastaFile(str(self.path))
            region, _sample = map(int, np.unravel_index(index, self.shape))
            contig, start, end = self.bed.row(region)
            seq = np.frombuffer(
                self.fasta.fetch(contig, start, end).encode("ascii").upper(), dtype="S1"
            )
            return seq.view("u1").astype(np.uint8, copy=True)

    return Ref(fasta, bed, n_samples)


def _measure_and_write(
    f, *, ds, dataset, backend, threads, seqlen, batch_size, n_batches,
    num_workers, burn_in, replicates, time_limit_ns, min_batches,
):
    from torch.utils.data import DataLoader
    from _bench_common import measure_cell, mib_per_s

    for _ in range(replicates):
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        res = measure_cell(
            dl, burn_in=burn_in, n_batches=n_batches,
            time_limit_ns=time_limit_ns, min_batches=min_batches,
        )
        prefix = f"{dataset},{backend},none,{threads},{seqlen},{batch_size}"
        if res is None:
            f.write(f"{prefix},0,0,0,nan\n")
        else:
            tput = mib_per_s(res.total_bytes, res.duration_ns / 1e9) if res.duration_ns > 0 else float("nan")
            f.write(f"{prefix},{res.n_measured},{res.total_bytes},{res.duration_ns},{tput}\n")
        f.flush()


def run_kind(
    *, kind, threads, fasta, bigwig_table, bed_dir, grid_dir, seqlens,
    results, n_samples, mem_cap_bytes, burn_in, replicates, time_limit_s, min_batches,
):
    from _bench_common import THROUGHPUT_HEADER

    bytes_per_bp = 1 if kind == "fasta" else 4
    backend = "fasta" if kind == "fasta" else "pybigwig"
    dataset = "FASTA" if kind == "fasta" else "TCGA_ATAC"
    num_workers = max(0, threads - 1)
    prefetch_factor = 2
    time_limit_ns = int(time_limit_s * 1e9)

    results.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results.exists()
    with open(results, "a") as f:
        if write_header:
            f.write(THROUGHPUT_HEADER)
            f.flush()
        for seqlen in seqlens:
            grid_file = Path(grid_dir) / f"grid_{seqlen}.csv"
            bed = Path(bed_dir) / f"tile_{seqlen}.bed"
            for batch_size, n_batches in select_cells(grid_file, threads=threads):
                if not cell_fits(
                    batch_size=batch_size, seqlen=seqlen, bytes_per_bp=bytes_per_bp,
                    num_workers=num_workers, prefetch_factor=prefetch_factor,
                    mem_cap_bytes=mem_cap_bytes,
                ):
                    f.write(f"{dataset},{backend},none,{threads},{seqlen},{batch_size},0,0,0,nan\n")
                    f.flush()
                    continue
                if kind == "fasta":
                    ds = _make_ref_dataset(fasta, bed, n_samples)
                else:
                    ds = _make_bigwig_dataset(bigwig_table, bed)
                _measure_and_write(
                    f, ds=ds, dataset=dataset, backend=backend, threads=threads,
                    seqlen=seqlen, batch_size=batch_size, n_batches=n_batches,
                    num_workers=num_workers, burn_in=burn_in, replicates=replicates,
                    time_limit_ns=time_limit_ns, min_batches=min_batches,
                )


def main(
    kind: str,
    threads: int,
    results: Path,
    *,
    fasta: Path = Path("/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa"),
    bigwig_table: Path = Path("/carter/shared/data/ml4gland/tcga-atac/data/sample_to_bigwig.csv"),
    bed_dir: Path = Path(__file__).parent / "beds",
    grid_dir: Path = Path(__file__).parent / "beds",
    n_samples: int = 62,
    mem_cap_gib: float = 96.0,
    burn_in: int = 1,
    replicates: int = 3,
    time_limit_s: float = 45.0,
    min_batches: int = 5,
):
    """Measure one baseline kind at one thread count. Threads are limited by the
    caller via taskset; this just reports `threads` and sets num_workers=threads-1."""
    if kind not in ("fasta", "pybigwig"):
        raise ValueError(f"kind must be 'fasta' or 'pybigwig', got {kind!r}")
    seqlens = [2048, 16384, 131072, 1048576]
    run_kind(
        kind=kind, threads=threads, fasta=fasta,
        bigwig_table=bigwig_table if kind == "pybigwig" else None,
        bed_dir=bed_dir, grid_dir=grid_dir, seqlens=seqlens, results=results,
        n_samples=n_samples, mem_cap_bytes=int(mem_cap_gib * 2**30),
        burn_in=burn_in, replicates=replicates, time_limit_s=time_limit_s, min_batches=min_batches,
    )


if __name__ == "__main__":
    run(main)
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `pixi run -e bench027 pytest hap_track_throughput/bin/tests/test_benchmark_baseline.py::test_fasta_driver_writes_comparable_schema -q`
Expected: PASS. (Requires `samtools` on PATH; it ships with the pysam conda package's deps. If absent, build the `.fai` with `pysam.faidx(str(fa))` instead.)

- [ ] **Step 5: Commit**

```bash
git add hap_track_throughput/bin/benchmark_baseline.py hap_track_throughput/bin/tests/test_benchmark_baseline.py
git commit -m "feat(baseline): FASTA driver via measure_cell + smoke test"
```

---

## Task 4: pyBigWig driver (`--kind pybigwig`) + smoke test

**Files:**
- Modify: `hap_track_throughput/bin/benchmark_baseline.py`
- Test: `hap_track_throughput/bin/tests/test_benchmark_baseline.py`

- [ ] **Step 1: Write the failing smoke test**

Append:
```python
def test_bigwig_driver_writes_comparable_schema(tmp_path):
    import numpy as np
    import polars as pl
    import pyBigWig
    from benchmark_baseline import run_kind

    # one tiny bigwig + a sample_to_bigwig table
    bw_path = tmp_path / "s1.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", 10_000)])
    bw.addEntries("chr1", 0, values=np.ones(10_000, dtype=np.float32), span=1, step=1)
    bw.close()
    table = tmp_path / "table.csv"
    pl.DataFrame({"sample": ["s1"], "path": [str(bw_path)]}).write_csv(table)

    beds = _write_bed(tmp_path, 2048, 4)  # noqa: F841 (writes tile_2048.bed)
    pl.DataFrame({"threads": [1], "batch_size": [2], "n_batches": [3]}).write_csv(tmp_path / "grid_2048.csv")
    out = tmp_path / "pybigwig.csv"

    run_kind(
        kind="pybigwig", threads=1, fasta=None, bigwig_table=table,
        bed_dir=tmp_path, grid_dir=tmp_path, seqlens=[2048],
        results=out, n_samples=1, mem_cap_bytes=96 * 2**30,
        burn_in=1, replicates=1, time_limit_s=5.0, min_batches=1,
    )
    df = pl.read_csv(out)
    assert (df["backend"] == "pybigwig").all()
    assert df["throughput (MiB/s)"].drop_nulls().gt(0).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e bench027 pytest hap_track_throughput/bin/tests/test_benchmark_baseline.py::test_bigwig_driver_writes_comparable_schema -q`
Expected: FAIL — `NameError: name '_make_bigwig_dataset' is not defined`.

- [ ] **Step 3: Implement `_make_bigwig_dataset`**

Insert into `benchmark_baseline.py` (before `run_kind`). The `BigWigDataset` body is copied verbatim from `benchmark_bigwig.py`; the `from_table` signature matches that script (`gvl.BigWigs.from_table("bw", table)`):
```python
def _make_bigwig_dataset(bigwig_table, bed: Path):
    import genvarloader as gvl
    import numpy as np
    import polars as pl
    from torch.utils.data import Dataset

    class BigWigDataset(Dataset):
        def __init__(self, bigwigs, bed):
            self.bigwigs = bigwigs
            self.bed = pl.read_csv(
                bed, separator="\t", has_header=False,
                new_columns=["contig", "start", "end"],
                schema_overrides={"contig": pl.Utf8},
            )
            self.n_samples = len(self.bigwigs.samples)

        @property
        def shape(self):
            return (self.bed.height, self.n_samples)

        def __len__(self):
            return self.bed.height * self.n_samples

        def __getitem__(self, index):
            region, sample = map(int, np.unravel_index(index, self.shape))
            contig, start, end = self.bed.row(region)
            return self.bigwigs.read(contig, start, end, sample=self.bigwigs.samples[sample])

    bigwigs = gvl.BigWigs.from_table("bw", bigwig_table)
    return BigWigDataset(bigwigs, bed)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e bench027 pytest hap_track_throughput/bin/tests/test_benchmark_baseline.py -q`
Expected: all tests pass (7 total).

- [ ] **Step 5: Commit**

```bash
git add hap_track_throughput/bin/benchmark_baseline.py hap_track_throughput/bin/tests/test_benchmark_baseline.py
git commit -m "feat(baseline): pyBigWig driver via measure_cell + smoke test"
```

---

## Task 5: Generate grids + tile BEDs, and validate grid parity with the 0.27 CSVs

**Files:**
- Create (generated, gitignored outputs ok): `hap_track_throughput/bin/beds/grid_{2048,16384,131072,1048576}.csv`, `hap_track_throughput/bin/beds/tile_{...}.bed`
- Test: `hap_track_throughput/bin/tests/test_grid_parity.py`

- [ ] **Step 1: Write the failing parity test**

```python
# hap_track_throughput/bin/tests/test_grid_parity.py
import sys
from pathlib import Path

import polars as pl

BIN = Path(__file__).resolve().parents[1]
ROOT = BIN.parents[1]
sys.path.insert(0, str(BIN))


def _gvl_cells(glob):
    frames = [pl.read_csv(p) for p in (ROOT / "results_gvl027").glob(glob)]
    df = pl.concat(frames)
    return set(
        (int(t), int(s), int(b))
        for t, s, b in df.select("threads", "seqlen", "batch_size").unique().iter_rows()
    )


def test_generated_grid_matches_gvl_haps_cells():
    """make_launch_grid.py defaults must reproduce the exact (threads,seqlen,batch) cells the 0.27 haps grid ran."""
    grid_cells = set()
    for seqlen in (2048, 16384, 131072, 1048576):
        g = pl.read_csv(BIN / "beds" / f"grid_{seqlen}.csv")
        for t, b in g.select("threads", "batch_size").unique().iter_rows():
            grid_cells.add((int(t), int(seqlen), int(b)))
    assert grid_cells == _gvl_cells("haps/*_none.csv")
```

- [ ] **Step 2: Generate the grids and beds, then run the test**

Run:
```bash
cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput
FASTA=/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa
mkdir -p bin/beds
for L in 2048 16384 131072 1048576; do
  pixi run -e bench027 python bin/make_launch_grid.py $L --output bin/beds/grid_$L.csv
  pixi run -e bench027 python bin/make_bed.py $L $FASTA bin/beds/tile_$L.bed
done
pixi run -e bench027 pytest bin/tests/test_grid_parity.py -q
```
Expected: PASS. If it FAILS, the 0.27 run used non-default `--max-npb`/`--min-npb`; inspect `results_gvl027/full_*.log` for the `make_launch_grid.py` invocation and pass the same flags, then re-run.

- [ ] **Step 3: Commit the generated grids (beds are large — gitignore the FASTA tiles, keep grids)**

```bash
git add hap_track_throughput/bin/beds/grid_*.csv hap_track_throughput/bin/tests/test_grid_parity.py
echo "beds/tile_*.bed" >> hap_track_throughput/bin/.gitignore
git add hap_track_throughput/bin/.gitignore
git commit -m "test(baseline): grid parity vs 0.27 haps cells; generated grids"
```

---

## Task 6: cn-03 driver sbatch + dry-run timing estimate

**Files:**
- Create: `hap_track_throughput/run_baselines.sbatch`

- [ ] **Step 1: Write the sbatch**

```bash
#!/bin/bash
#SBATCH --job-name=gvl027-baselines
#SBATCH --partition=carter-compute
#SBATCH --account=carter
#SBATCH --nodelist=carter-cn-03
#SBATCH --exclusive
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/carter/users/dlaub/projects/gvl-paper/results_gvl027/baselines_%j.log
#SBATCH --error=/carter/users/dlaub/projects/gvl-paper/results_gvl027/baselines_%j.log
#
# Re-measure FASTA + pyBigWig baselines apples-to-apples with the GVL 0.27 grid.
# Pinned to cn-03 (same node as the 0.27 grid; cn-02 is bandwidth-starved).
# FASTA then pyBigWig, SEQUENTIALLY — two throughput benchmarks on one node would
# contend for DRAM bandwidth and corrupt each other's numbers. Thread counts are
# limited per-cell with taskset so each cell uses exactly T cores (num_workers=T-1).
set -euo pipefail
export PATH="/cellar/users/dlaub/.pixi/bin:$PATH"
ROOT=/carter/users/dlaub/projects/gvl-paper
cd "$ROOT/hap_track_throughput"

OUT="$ROOT/results_gvl027/baselines"
mkdir -p "$OUT"
THREADS=(1 4 16 32)

for KIND in fasta pybigwig; do
  RES="$OUT/${KIND}.csv"
  rm -f "$RES"   # fresh; the script appends across thread counts
  for T in "${THREADS[@]}"; do
    echo "=== $KIND threads=$T $(date) ==="
    taskset -c 0-$((T-1)) pixi run -e bench027 python bin/benchmark_baseline.py \
      "$KIND" "$T" "$RES"
  done
done
echo "=== DONE $(date) ==="
```

- [ ] **Step 2: Dry-run timing estimate (do NOT launch the full job yet)**

Run a single cheap thread count on cn-03 interactively or as a short job and extrapolate:
```bash
cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput
# one thread count, fasta only -> time it, then x4 threads x2 kinds is the upper bound
/usr/bin/time -v taskset -c 0-0 pixi run -e bench027 python bin/benchmark_baseline.py \
  fasta 1 /tmp/baseline_dryrun.csv 2>&1 | tail -20
```
Expected: completes; note wall time `W`. Full run upper bound ≈ `W × (1+4+16+32 thread-cost factor)` — but threads run sequentially and larger-thread cells are faster, so total ≈ a few × `W`. If the projection exceeds ~6 h, lower `time_limit_s` (e.g. `--time-limit-s 30`) or `replicates` (`--replicates 2`) and note it in the run log. Record the estimate in the commit message.

- [ ] **Step 3: Commit**

```bash
git add hap_track_throughput/run_baselines.sbatch
git commit -m "feat(baseline): cn-03 sequential driver sbatch (taskset thread sweep)"
```

---

## Task 7: Speedup + A100 analysis

**Files:**
- Create: `hap_track_throughput/bin_gvl027/compute_speedups.py`

- [ ] **Step 1: Write the analysis script**

```python
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
        frames = [pl.read_csv(p) for p in (results_dir).glob(glob)]
        df = pl.concat(frames).filter(pl.col("throughput (MiB/s)").is_not_nan())
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
```

- [ ] **Step 2: Run after the bench completes (placeholder run now to check it parses)**

Run: `pixi run -e default python hap_track_throughput/bin_gvl027/compute_speedups.py --help`
Expected: cyclopts help prints (no crash). Full run happens once `results_gvl027/baselines/{fasta,pybigwig}.csv` exist.

- [ ] **Step 3: Commit**

```bash
git add hap_track_throughput/bin_gvl027/compute_speedups.py
git commit -m "feat(baseline): compute_speedups.py (GVL/baseline ratios + A100 check)"
```

---

## Task 8: Launch + record results (manual gate)

**Files:** none (run + record).

- [ ] **Step 1: Submit on cn-03**

Run: `cd /carter/users/dlaub/projects/gvl-paper/hap_track_throughput && sbatch run_baselines.sbatch`
Then watch: `squeue -u $USER` and the log under `results_gvl027/baselines_*.log`.

- [ ] **Step 2: After completion, compute speedups**

Run: `pixi run -e default python hap_track_throughput/bin_gvl027/compute_speedups.py`
Expected: `results_gvl027/speedups.csv` written; note the speedup range and the A100 verdict — these feed the manuscript re-basing (roadmap §0) and the GPU-bandwidth softening.

- [ ] **Step 3: Record outcomes in the roadmap (manuscript repo)**

Update `text/roadmap.md §0` with the measured speedup ranges and A100 verdict; commit in the local `text/` repo.

---

## Self-review notes
- **Spec coverage:** measure_cell reuse (T2–4), num_workers=n_cpus-1 (T3 `run_kind`), grid parity (T5), memory guard (T2/T3), cn-03 sequential + taskset thread sweep (T6), single FASTA curve + tcga-atac pyBigWig (T3/T4/T7), schema parity (T3 test), <6h check (T6 dry-run), compare/A100 (T7). All covered.
- **Env:** `bench027` (T1) for everything except `compute_speedups.py`, which only reads CSVs + needs polars → runs in `default` (matches `compare_to_baseline.py`'s env choice).
- **Threads vs num_workers:** swept by taskset (one invocation per thread count); `num_workers = threads-1` derived from the same `threads` arg — consistent across T3/T6.
