"""Pair-stream helpers shared by generate_pairs.py and the bench_*.py scripts."""

import polars as pl

Pair = tuple[tuple[str, int, int], str]


def compute_batch_size(query_length: int, bp_budget: int) -> int:
    """Pairs per batch holding base-pairs-per-batch approximately constant.

    Mirrors the convention used by the memory-growth figure: a batch covers at
    most `bp_budget` base pairs, so batch size scales inversely with query length.
    """
    return max(1, bp_budget // query_length)


def split_pair_batches(group: pl.DataFrame) -> list[list[Pair]]:
    """Split one replicate's rows into batches by `batch_id` (ascending).

    Each pair is ((contig, start, end), sample). Row order within a batch is
    preserved.
    """
    batches: list[list[Pair]] = []
    for _, batch in group.sort("batch_id").group_by("batch_id", maintain_order=True):
        batches.append(
            [
                ((row["contig"], int(row["start"]), int(row["end"])), row["sample"])
                for row in batch.iter_rows(named=True)
            ]
        )
    return batches
