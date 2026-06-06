import numpy as np
import pytest

from _bench_common import (
    MEMORY_HEADER,
    THROUGHPUT_HEADER,
    CellResult,
    mib_per_s,
    n_bytes,
    measure_cell,
)


def test_headers_match_reconciled_schema():
    assert THROUGHPUT_HEADER == (
        "dataset,backend,dl_mode,threads,seqlen,batch_size,"
        "n_batches_measured,total_bytes,duration_ns,throughput (MiB/s)\n"
    )
    assert MEMORY_HEADER == (
        "dataset,backend,dl_mode,threads,seqlen,batch_size,"
        "avg_rss_bytes,peak_rss_bytes\n"
    )


def test_n_bytes_numpy():
    arr = np.zeros((4, 2048), dtype="S1")
    assert n_bytes(arr) == 4 * 2048 * 1
    farr = np.zeros((4, 2048), dtype=np.float32)
    assert n_bytes(farr) == 4 * 2048 * 4


def test_n_bytes_torch_like():
    # torch.Tensor exposes numel()/element_size() and ALSO a .size method +
    # .itemsize attr; the numpy-first dispatch must take the torch branch.
    class FakeTensor:
        itemsize = 4

        def numel(self):
            return 8

        def element_size(self):
            return 4

        def size(self):
            return (2, 4)

    assert n_bytes(FakeTensor()) == 8 * 4


def test_mib_per_s():
    assert mib_per_s(2**20, 1.0) == pytest.approx(1.0)


class _FakeBatch:
    """Minimal numpy-like: n_bytes uses .size and .itemsize."""

    def __init__(self, nbytes: int):
        self.size = nbytes
        self.itemsize = 1


def _clock():
    """Deterministic ns clock advancing 1 ns per call."""
    t = {"n": 0}

    def now() -> int:
        t["n"] += 1
        return t["n"]

    return now


def test_measure_cell_counts_bytes_after_burn_in():
    # 3 batches of 100 bytes; burn_in=1 -> bytes counted for batches with
    # n_yielded >= burn_in (i.e. all 3 here, since burn_in index == first batch).
    dl = [_FakeBatch(100), _FakeBatch(100), _FakeBatch(100)]
    res = measure_cell(
        dl, burn_in=1, n_batches=3, time_limit_ns=10**18, min_batches=1,
        now_ns=_clock(),
    )
    assert isinstance(res, CellResult)
    # batches with n_yielded >= burn_in(=1): all three (n_yielded 1,2,3)
    assert res.total_bytes == 300
    # measured = batches with n_yielded > burn_in handling: see impl; here 3 counted
    assert res.n_measured == 3
    assert res.duration_ns > 0


def test_measure_cell_empty_epoch_returns_none():
    # An epoch that yields zero batches must NOT spin forever; returns None (NaN).
    res = measure_cell(
        [], burn_in=1, n_batches=5, time_limit_ns=10**18, min_batches=1,
        now_ns=_clock(),
    )
    assert res is None


def test_measure_cell_reiterates_until_n_batches():
    # A 2-batch loader, asked for 4 measured batches, must re-iterate (epoch twice).
    dl = [_FakeBatch(10), _FakeBatch(10)]
    res = measure_cell(
        dl, burn_in=0, n_batches=4, time_limit_ns=10**18, min_batches=1,
        now_ns=_clock(),
    )
    assert res is not None
    assert res.n_measured == 4
    assert res.total_bytes == 40
