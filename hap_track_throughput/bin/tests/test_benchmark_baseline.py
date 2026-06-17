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
