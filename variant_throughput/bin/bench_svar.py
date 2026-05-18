#! /usr/bin/env python

from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

import awkward as ak
import numpy as np
from awkward.contents import Content
from cyclopts import run
from genoray import SparseVar
from numba import njit, prange
from numpy.typing import NDArray


@njit(parallel=True, nogil=True, cache=True)
def _gather_parallel(
    src: NDArray[np.int32],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    dst: NDArray[np.int32],
    dst_offsets: NDArray[np.int64],
) -> None:
    for i in prange(len(starts)):
        i_s = starts[i]
        i_e = ends[i]
        o_s, o_e = dst_offsets[i], dst_offsets[i + 1]
        dst[o_s:o_e] = src[i_s:i_e]


def to_packed_custom(node: Content) -> Content:
    from awkward.contents import ListArray, ListOffsetArray, NumpyArray, RegularArray
    from awkward.index import Index

    reg_sizes: list[int] = []
    inner = node
    while isinstance(inner, RegularArray):
        reg_sizes.append(inner.size)
        inner = inner.content

    if not isinstance(inner, ListArray):
        return ak.to_packed(node)

    starts = np.asarray(inner.starts.data, dtype=np.int64)
    ends = np.asarray(inner.stops.data, dtype=np.int64)
    lengths = ends - starts
    dst_offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    dst_offsets[0] = 0
    np.cumsum(lengths, out=dst_offsets[1:])

    src = np.asarray(inner.content.data)
    dst = np.empty(int(dst_offsets[-1]), dtype=src.dtype)
    _gather_parallel(src, starts, ends, dst, dst_offsets)

    packed = ListOffsetArray(Index(dst_offsets), NumpyArray(dst))
    for size in reversed(reg_sizes):
        packed = RegularArray(packed, size)
    return packed


def _svar_search(
    _svar: SparseVar,
    pairs: list[tuple[tuple[str, int, int], str]],
) -> tuple[NDArray, NDArray, int]:
    """Vectorized index search. Returns flat_starts, flat_ends, n_calls."""

    by_contig: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    for i, ((c, s, e), smp) in enumerate(pairs):
        by_contig[c].append((i, s, e, smp))

    start_offs: list[NDArray] = [None] * len(pairs)  # type: ignore[list-item]
    end_offs: list[NDArray] = [None] * len(pairs)  # type: ignore[list-item]

    for contig, rows in by_contig.items():
        _, starts_list, ends_list, samples_list = zip(*rows)
        unique_samples = list(set(samples_list))
        sample_idx = {s: j for j, s in enumerate(unique_samples)}

        offs = _svar._find_starts_ends(
            contig,
            np.asarray(starts_list, dtype=np.int64),
            np.asarray(ends_list, dtype=np.int64),
            np.asarray(unique_samples),
        )

        for r, (i, _, _, smp) in enumerate(rows):
            j = sample_idx[smp]
            start_offs[i] = offs[0, r, j, :]
            end_offs[i] = offs[1, r, j, :]

    flat_starts = np.concatenate(start_offs)
    flat_ends = np.concatenate(end_offs)
    n_calls = int((flat_ends - flat_starts).sum())
    return flat_starts, flat_ends, n_calls


def _svar_pack(
    _svar: SparseVar,
    flat_starts: NDArray,
    flat_ends: NDArray,
    use_custom_pack: bool,
) -> None:
    """Gather (read) step — the timed operation for throughput."""
    from awkward.contents import ListArray, NumpyArray, RegularArray
    from awkward.index import Index

    node = NumpyArray(_svar.genos.data)  # type: ignore[arg-type]
    node = ListArray(Index(flat_starts), Index(flat_ends), node)
    node = RegularArray(node, 2)
    pack_fn = to_packed_custom if use_custom_pack else ak.to_packed
    pack_fn(node)


def bench(
    pairs_parquet: Path,
    svar: Path,
    output: Path,
    dataset: str = "",
    mode: Literal["throughput", "memory"] = "throughput",
    use_custom_pack: bool = True,
    n_samples: int = 0,
):
    import polars as pl

    _svar = SparseVar(svar)

    # Warm up numba JIT with the real memmap dtype before any timing.
    _w_src = np.zeros(4, dtype=np.int32)
    _w_src.flags.writeable = False
    _w_dst = np.zeros(4, dtype=np.int32)
    _w_starts = np.array([0, 2], dtype=np.int64)
    _w_ends = np.array([2, 4], dtype=np.int64)
    _w_offsets = np.array([0, 2, 4], dtype=np.int64)
    _gather_parallel(_w_src, _w_starts, _w_ends, _w_dst, _w_offsets)

    df = pl.read_parquet(pairs_parquet)
    q_len = int(df["end"][0] - df["start"][0])

    rows_out: list[dict] = []

    for rep_val, group in df.group_by("replicate", maintain_order=True):
        rep = rep_val[0] if isinstance(rep_val, tuple) else rep_val
        pairs = [
            ((row["contig"], int(row["start"]), int(row["end"])), row["sample"])
            for row in group.iter_rows(named=True)
        ]
        if not pairs:
            continue

        if mode == "throughput":
            t0 = perf_counter_ns()
            flat_starts, flat_ends, n_calls = _svar_search(_svar, pairs)
            search_ns = perf_counter_ns() - t0

            t0 = perf_counter_ns()
            _svar_pack(_svar, flat_starts, flat_ends, use_custom_pack)
            pack_ns = perf_counter_ns() - t0

            rows_out.append({
                "dataset": dataset or svar.name,
                "method": "svar",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": len(pairs),
                "n_calls": n_calls,
                "elapsed_ns": pack_ns,
                "setup_ns": search_ns,
            })
        else:
            from _mem_sampler import PeakRssSampler

            flat_starts, flat_ends, n_calls = _svar_search(_svar, pairs)
            with PeakRssSampler() as s:
                _svar_pack(_svar, flat_starts, flat_ends, use_custom_pack)

            rows_out.append({
                "dataset": dataset or svar.name,
                "method": "svar",
                "query_length": q_len,
                "n_samples": int(n_samples),
                "replicate": int(rep),
                "n_pairs": len(pairs),
                "n_calls": n_calls,
                "peak_rss_bytes": s.peak,
            })

    pl.DataFrame(rows_out).write_csv(output)


if __name__ == "__main__":
    run(bench)
