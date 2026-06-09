import polars as pl

from _pairs import compute_batch_size, split_pair_batches


def test_compute_batch_size_inverse_scaling():
    assert compute_batch_size(2048, 2**24) == 8192
    assert compute_batch_size(2**24, 2**24) == 1
    # never below 1, even when query_length exceeds the budget
    assert compute_batch_size(2**25, 2**24) == 1


def test_split_pair_batches_groups_by_batch_id_in_order():
    df = pl.DataFrame(
        {
            "batch_id": [0, 0, 1],
            "contig": ["1", "1", "2"],
            "start": [10, 20, 30],
            "end": [12, 22, 32],
            "sample": ["s0", "s1", "s2"],
        }
    )
    batches = split_pair_batches(df)
    assert batches == [
        [(("1", 10, 12), "s0"), (("1", 20, 22), "s1")],
        [(("2", 30, 32), "s2")],
    ]
