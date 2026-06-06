#! /usr/bin/env python


import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "bin"))

import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import awkward as ak
import numpy as np
from awkward.contents import Content
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
    """Numba-parallel gather for ListArray (optionally wrapped in RegularArray).
    Falls through to ak.to_packed() for ListOffsetArray content."""
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


def load_gap_intervals(
    contig_map: dict[str, str],
    known_contigs: set[str],
    cache_dir: Path | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Download (once) and parse UCSC hg38 gap table into per-contig gap intervals."""
    import urllib.request

    import polars as pl

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "gvl-paper"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "hg38_gap.txt.gz"
    if not cache_path.exists():
        url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz"
        print(f"Downloading UCSC gap table to {cache_path} ...")
        urllib.request.urlretrieve(url, cache_path)

    df = (
        pl.read_csv(
            cache_path,
            separator="\t",
            has_header=False,
            new_columns=[
                "bin",
                "chrom",
                "start",
                "end",
                "ix",
                "n",
                "size",
                "type",
                "bridge",
            ],
        )
        .select(
            pl.col("chrom").replace_strict(contig_map, default=None).alias("contig"),
            "start",
            "end",
        )
        .drop_nulls()
        .filter(pl.col("contig").is_in(list(known_contigs)))
        .sort(["contig", "start"])
    )

    result: dict[str, list[tuple[int, int]]] = {c: [] for c in known_contigs}
    for row in df.iter_rows(named=True):
        result[row["contig"]].append((row["start"], row["end"]))
    return result


def build_allowed_intervals(
    contig_lengths: dict[str, int],
    gap_intervals: dict[str, list[tuple[int, int]]],
) -> list[tuple[str, int, int]]:
    """Complement gap intervals within each contig to get allowed (non-gap) intervals."""
    allowed: list[tuple[str, int, int]] = []
    for contig, length in contig_lengths.items():
        gaps = sorted(gap_intervals.get(contig, []))
        pos = 0
        for gap_start, gap_end in gaps:
            if gap_start > pos:
                allowed.append((contig, pos, gap_start))
            pos = max(pos, gap_end)
        if pos < length:
            allowed.append((contig, pos, length))
    return allowed


def sample_region(
    allowed: list[tuple[str, int, int]],
    q_len: int,
    rng: random.Random,
) -> tuple[str, int, int] | None:
    """Sample a uniformly random start position in non-gap intervals with length >= q_len."""
    eligible = [(c, s, e) for c, s, e in allowed if e - s >= q_len]
    if not eligible:
        return None
    weights = [e - s - q_len + 1 for _, s, e in eligible]
    (contig, ivl_start, ivl_end) = rng.choices(eligible, weights=weights, k=1)[0]
    start = rng.randint(ivl_start, ivl_end - q_len)
    return contig, start, start + q_len


def generate_batch(
    allowed: list[tuple[str, int, int]],
    q_len: int,
    available_samples: list[str],
    rng: random.Random,
    max_pairs: int = 100,
    max_total_length: int = 2**24,
) -> list[tuple[tuple[str, int, int], str]]:
    """Sample a random batch of (region, sample) pairs of fixed query length."""
    target = rng.randint(1, max_pairs)
    pairs: list[tuple[tuple[str, int, int], str]] = []
    total = 0
    for _ in range(target):
        if total + q_len > max_total_length:
            break
        region = sample_region(allowed, q_len, rng)
        if region is None:
            break
        sample = rng.choice(available_samples)
        pairs.append((region, sample))
        total += q_len
    return pairs


def bench_svar_batch(
    _svar, pairs: list[tuple[tuple[str, int, int], str]], use_custom_pack: bool = True
) -> tuple[int, int, int]:
    """Time the full SVAR batch path, returning (svar_ns, search_ns, n_calls, pack_custom_ns).

    Groups pairs by contig and calls _find_starts_ends once per contig (vectorized).
    search_ns accumulates only time inside _find_starts_ends; svar_ns is the full
    pipeline including awkward array construction and to_packed_custom().
    """
    from awkward.contents import ListArray, NumpyArray, RegularArray
    from awkward.index import Index

    search_ns = 0

    by_contig: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    for i, ((c, s, e), smp) in enumerate(pairs):
        by_contig[c].append((i, s, e, smp))

    start_offs: list[NDArray] = [None] * len(pairs)  # type: ignore[list-item]
    end_offs: list[NDArray] = [None] * len(pairs)  # type: ignore[list-item]

    for contig, rows in by_contig.items():
        _, starts, ends, samples_list = zip(*rows)
        unique_samples = list(set(samples_list))
        sample_idx = {s: j for j, s in enumerate(unique_samples)}

        t_s = perf_counter_ns()
        offs = _svar._find_starts_ends(
            contig,
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
            np.asarray(unique_samples),
        )  # (2, n_ranges, n_unique_samples, ploidy)
        search_ns += perf_counter_ns() - t_s

        for r, (i, _, _, smp) in enumerate(rows):
            j = sample_idx[smp]
            start_offs[i] = offs[0, r, j, :]
            end_offs[i] = offs[1, r, j, :]

    flat_starts = np.concatenate(start_offs)  # (n_pairs * ploidy,)
    flat_ends = np.concatenate(end_offs)
    n_calls = int((flat_ends - flat_starts).sum())

    node = NumpyArray(_svar.genos.data)  # type: ignore[arg-type]
    node = ListArray(Index(flat_starts), Index(flat_ends), node)
    node = RegularArray(node, 2)  # (n_pairs, 2 ploidy, ~variants)
    pack_fn = to_packed_custom if use_custom_pack else ak.to_packed
    t0 = perf_counter_ns()
    pack_fn(node)
    pack_time = perf_counter_ns() - t0

    return pack_time, search_ns, n_calls


def bench_presubset_bcf_batch(
    bcf: Path,
    pairs: list[tuple[tuple[str, int, int], str]],
    tmp_dir: Path | None = None,
) -> tuple[int, int, int]:
    """Subset BCF to each (region, sample) with bcftools, then read all with cyvcf2.

    Returns (subset_ns, read_ns, n_variants). subset_ns covers all bcftools writes;
    read_ns covers sequential cyvcf2 iteration over all subset BCFs. Cleanup is
    unconditional via try/finally and is not timed.
    """
    import cyvcf2

    if tmp_dir is None:
        tmp_dir = Path(__file__).parent / ".bench_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_paths: list[str] = []
    try:
        t0 = perf_counter_ns()
        for (contig, start, end), sample in pairs:
            fd, path = tempfile.mkstemp(suffix=".bcf", dir=tmp_dir)
            os.close(fd)
            tmp_paths.append(path)
            subprocess.run(
                [
                    "bcftools",
                    "view",
                    "-s",
                    sample,
                    "-r",
                    f"{contig}:{start + 1}-{end}",
                    "--min-ac",
                    "1",
                    "--no-update",
                    "-Ob",
                    "-o",
                    path,
                    str(bcf),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        subset_ns = perf_counter_ns() - t0

        t0 = perf_counter_ns()
        total_v = 0
        for path in tmp_paths:
            vcf = cyvcf2.VCF(path, lazy=True)
            chunks: list[NDArray] = []
            for v in vcf:
                chunks.append(v.genotype.array())  # (1, ploidy+1) per variant
            if chunks:
                np.concatenate(chunks, axis=0)  # (n_variants, ploidy+1); not retained
            total_v += len(chunks)
            vcf.close()
        read_ns = perf_counter_ns() - t0
    finally:
        for path in tmp_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    return subset_ns, read_ns, total_v


def main(
    svar: Path,
    bcf: Path,
    pgen: Path,
    fai: Path,
    results: Path,
    n_replicates: int = 5,
    max_pairs: int = 100,
    max_total_length: int = 2**24,
    query_lengths: list[int] | None = None,
    seed: int = 0,
    tmp_dir: Path | None = None,
    use_custom_pack: bool = True,
):
    import polars as pl
    from genoray import PGEN, VCF, SparseVar
    from rich.progress import MofNCompleteColumn, Progress

    if query_lengths is None:
        query_lengths = (2 ** np.arange(11, 25)).tolist()

    from _genoray_filter import hap_safe_pl_filter, hap_safe_vcf_callable

    _hap_pl = hap_safe_pl_filter()
    _svar = SparseVar(svar)
    _bcf = VCF(bcf, with_gvi_index=False, filter=hap_safe_vcf_callable(), pl_filter=_hap_pl)
    _pgen = PGEN(pgen, filter=_hap_pl)

    _fai = (
        pl.read_csv(
            fai,
            separator="\t",
            has_header=False,
            new_columns=["contig", "length"],
        )
        .select(
            pl.col("contig").replace_strict(_svar._c_norm.contig_map, default=None),
            "length",
        )
        .drop_nulls()
        .filter(pl.col("contig").is_in(_svar.contigs))
    )
    contig_lengths: dict[str, int] = {
        c: len_ for c, len_ in zip(*_fai.get_columns()) if c in _svar.contigs
    }

    gap_intervals = load_gap_intervals(
        contig_map=_svar._c_norm.contig_map,
        known_contigs=set(contig_lengths),
    )
    allowed = build_allowed_intervals(contig_lengths, gap_intervals)

    total_allowed = sum(e - s for _, s, e in allowed)
    total_contig = sum(contig_lengths.values())
    print(
        f"Allowed (non-gap) bp: {total_allowed:,} / {total_contig:,} ({100 * total_allowed / total_contig:.1f}%)"
    )

    available_samples = list(_svar.available_samples)
    rng = random.Random(seed)

    # Trigger numba JIT compilation with the real memmap dtype (int32) before any timing.
    _w_src = np.zeros(4, dtype=np.int32)
    _w_src.flags.writeable = False
    _w_dst = np.zeros(4, dtype=np.int32)
    _w_starts = np.array([0, 2], dtype=np.int64)
    _w_ends = np.array([2, 4], dtype=np.int64)
    _w_offsets = np.array([0, 2, 4], dtype=np.int64)
    _gather_parallel(_w_src, _w_starts, _w_ends, _w_dst, _w_offsets)

    rec_query_length: list[int] = []
    rec_replicate: list[int] = []
    rec_n_pairs: list[int] = []
    rec_n_calls: list[int] = []
    rec_n_variants: list[int] = []
    svar_search_times: list[int] = []
    svar_times: list[int] = []
    bcf_times: list[int] = []
    plink_times: list[int] = []
    bcf_subset_times: list[int] = []
    presubset_bcf_times: list[int] = []

    pbar = Progress(*Progress.get_default_columns(), MofNCompleteColumn())
    pbar.start()
    task = pbar.add_task("Benchmarking", total=len(query_lengths))

    for q_len in query_lengths:
        for rep in range(n_replicates):
            pairs = generate_batch(
                allowed,
                q_len,
                available_samples,
                rng,
                max_pairs=max_pairs,
                max_total_length=max_total_length,
            )
            if not pairs:
                continue

            rec_query_length.append(q_len)
            rec_replicate.append(rep)
            rec_n_pairs.append(len(pairs))

            svar_ns, search_ns, n_calls = bench_svar_batch(
                _svar, pairs, use_custom_pack=use_custom_pack
            )
            svar_times.append(svar_ns)
            svar_search_times.append(search_ns)
            rec_n_calls.append(n_calls)

            t0 = perf_counter_ns()
            total_v = 0
            for (contig, start, end), sample in pairs:
                _bcf = _bcf.set_samples(sample)
                genos = _bcf.read(contig, start, end, mode=_bcf.Genos8)
                total_v += genos.shape[-1]
            bcf_times.append(perf_counter_ns() - t0)
            rec_n_variants.append(total_v)

            t0 = perf_counter_ns()
            for (contig, start, end), sample in pairs:
                _pgen = _pgen.set_samples(sample)
                _pgen.read(contig, start, end, mode=_pgen.Genos)
            plink_times.append(perf_counter_ns() - t0)

            sub_ns, pre_ns, _ = bench_presubset_bcf_batch(bcf, pairs, tmp_dir)
            bcf_subset_times.append(sub_ns)
            presubset_bcf_times.append(pre_ns)

        pbar.update(task, advance=1)

    pbar.stop()

    pl.DataFrame({
        "query_length": rec_query_length,
        "replicate": rec_replicate,
        "n_pairs": rec_n_pairs,
        "n_calls": rec_n_calls,
        "n_variants": rec_n_variants,
        "svar_search_time": svar_search_times,
        "svar_time": svar_times,
        "bcf_time": bcf_times,
        "plink_time": plink_times,
        "bcf_subset_time": bcf_subset_times,
        "presubset_bcf_time": presubset_bcf_times,
    }).write_csv(results)


if __name__ == "__main__":
    ddir = Path("/carter/users/dlaub/data/1kGP")
    res_dir = Path("/cellar/users/dlaub/projects/gvl-paper/results")

    main(
        svar=ddir / "1kGP.snp_indel.split_multiallelics.svar",
        bcf=ddir / "1kGP.snp_indel.split_multiallelics.bcf",
        pgen=ddir / "1kGP.snp_indel.split_multiallelics.pgen",
        fai=ddir / "GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
        results=res_dir / "variants_batched_throughput_1kgp_par4_nb_gather.csv",
        use_custom_pack=True,
    )
