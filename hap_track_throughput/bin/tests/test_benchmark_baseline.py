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


def _write_tiny_fasta(tmp_path) -> Path:
    import subprocess
    fa = tmp_path / "tiny.fa"
    # one contig "chr1", 10 kb of ACGT so 2048-bp reads have room
    seq = ("ACGT" * 2560)[:10_000]
    fa.write_text(">chr1\n" + "\n".join(seq[i:i+60] for i in range(0, len(seq), 60)) + "\n")
    try:
        subprocess.run(["samtools", "faidx", str(fa)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        import pysam
        pysam.faidx(str(fa))
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


def test_fasta_pads_chromosome_boundary_tile(tmp_path):
    """pysam clips tiles that overhang chromosome ends; __getitem__ must zero-pad to seqlen."""
    import numpy as np
    import pysam
    from benchmark_baseline import _make_ref_dataset

    fa = tmp_path / "short.fa"
    fa.write_text(">chr1\n" + "A" * 1500 + "\n")
    pysam.faidx(str(fa))
    bed = tmp_path / "tile_2048.bed"
    bed.write_text("chr1\t0\t2048\n")  # end (2048) > chrom_len (1500): pysam will clip

    ds = _make_ref_dataset(fa, bed, n_samples=1)
    item = ds[0]
    assert len(item) == 2048, f"expected 2048, got {len(item)}"
    assert item.dtype == np.uint8
    assert (item[1500:] == 0).all(), "tail must be zero-padded"
    assert (item[:1500] != 0).any(), "leading bases must be non-zero"
