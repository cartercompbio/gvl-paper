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

    # Mirror a real torch.Tensor: it has numel()/element_size() AND a .size
    # *method* plus an .itemsize attribute. The numpy-first dispatch would have
    # misrouted this (calling int(<bound method>) -> TypeError / wrong branch),
    # so this asserts the torch path is taken.
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
    # 2**20 bytes in 1 second == 1 MiB/s
    assert mib_per_s(2 ** 20, 1.0) == pytest.approx(1.0)
