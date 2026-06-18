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


def _make_ref_dataset(fasta: Path, fasta2: Path, bed: Path, n_samples: int, drop_cache: bool = True):
    import os
    import numpy as np
    import polars as pl
    import pysam
    from torch.utils.data import Dataset

    class Ref(Dataset):
        def __init__(self, path, path2, bed, n_samples, drop_cache):
            self.path = path
            self.path2 = path2
            # fasta and fasta2 handles opened lazily in __getitem__ (worker-fork-safe)
            self.fasta = None
            self.fasta2 = None
            # eviction fds opened lazily per-worker after fork (drop_cache path only)
            self._evict_fd1 = None
            self._evict_fd2 = None
            self.drop_cache = drop_cache
            self.bed = pl.read_csv(
                bed, separator="\t", has_header=False,
                new_columns=["contig", "start", "end"],
                schema_overrides={"contig": pl.Utf8},
            )
            self.n_samples = n_samples
            # Contig/UCSC reconciliation based on fasta (hap1).
            # fasta2 is a byte-identical duplicate so it has the same contig names.
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

        def _read_padded(self, handle, contig, start, end):
            """Fetch [start, end) from handle and zero-pad to seqlen if pysam clips."""
            seqlen = end - start
            raw = handle.fetch(contig, start, end).encode("ascii").upper()
            seq = np.frombuffer(raw, dtype="S1").view("u1").astype(np.uint8, copy=True)
            if len(seq) < seqlen:
                # Pad to expected length (chromosome boundary truncation — 25 tiles
                # in GRCh38 tile_2048.bed have end > chrom_len; pysam clips them).
                # Fill with 0 (same convention as GVL's N-masking).
                out = np.zeros(seqlen, dtype=np.uint8)
                out[: len(seq)] = seq
                return out
            return seq

        def __getitem__(self, index):
            if self.fasta is None:
                self.fasta = pysam.FastaFile(str(self.path))
            if self.fasta2 is None:
                self.fasta2 = pysam.FastaFile(str(self.path2))
            region, _sample = map(int, np.unravel_index(index, self.shape))
            contig, start, end = self.bed.row(region)
            h1 = self._read_padded(self.fasta, contig, start, end)
            h2 = self._read_padded(self.fasta2, contig, start, end)
            # Stack to (2, seqlen) matching GVL's diploid (2, seqlen) uint8 output.
            result = np.stack([h1, h2])
            if self.drop_cache:
                # Evict each haplotype file's pages from the OS page cache after every
                # read so the next item sees a cold storage read. This mirrors the
                # real bcftools-consensus workflow: 2*n_samples FASTAs (tens of TB)
                # never fit in RAM, so reads are always cold. GVL's small working set
                # legitimately fits in RAM — page-cache eviction is the honest baseline.
                # Lazily open separate fds for fadvise (per-worker after fork).
                if self._evict_fd1 is None:
                    self._evict_fd1 = os.open(str(self.path), os.O_RDONLY)
                if self._evict_fd2 is None:
                    self._evict_fd2 = os.open(str(self.path2), os.O_RDONLY)
                # offset=0, length=0 → entire file
                os.posix_fadvise(self._evict_fd1, 0, 0, os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(self._evict_fd2, 0, 0, os.POSIX_FADV_DONTNEED)
            return result

    return Ref(fasta, fasta2, bed, n_samples, drop_cache)


def _measure_and_write(
    f, *, ds, dataset, backend, threads, seqlen, batch_size, n_batches,
    num_workers, prefetch_factor, burn_in, replicates, time_limit_ns, min_batches,
):
    from torch.utils.data import DataLoader
    from _bench_common import measure_cell, mib_per_s

    for _ in range(replicates):
        dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor
        dl = DataLoader(ds, **dl_kwargs)
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


def run_kind(
    *, kind, threads, fasta, fasta2=None, bigwig_table, bed_dir, grid_dir, seqlens,
    results, n_samples, mem_cap_bytes, burn_in, replicates, time_limit_s, min_batches,
    drop_cache: bool = True,
):
    from _bench_common import THROUGHPUT_HEADER

    # fasta items are diploid (2, seqlen) uint8 = 2 bytes/bp, matching GVL haps output.
    bytes_per_bp = 2 if kind == "fasta" else 4
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
                    ds = _make_ref_dataset(fasta, fasta2, bed, n_samples, drop_cache=drop_cache)
                else:
                    ds = _make_bigwig_dataset(bigwig_table, bed)
                _measure_and_write(
                    f, ds=ds, dataset=dataset, backend=backend, threads=threads,
                    seqlen=seqlen, batch_size=batch_size, n_batches=n_batches,
                    num_workers=num_workers, prefetch_factor=prefetch_factor,
                    burn_in=burn_in, replicates=replicates,
                    time_limit_ns=time_limit_ns, min_batches=min_batches,
                )


def main(
    kind: str,
    threads: int,
    results: Path,
    *,
    fasta: Path = Path("/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa"),
    fasta2: Path = Path("/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.hap2.fa"),
    bigwig_table: Path = Path("/carter/shared/data/ml4gland/tcga-atac/data/sample_to_bigwig.csv"),
    bed_dir: Path = Path(__file__).parent / "beds",
    grid_dir: Path = Path(__file__).parent / "beds",
    n_samples: int = 62,
    mem_cap_gib: float = 96.0,
    burn_in: int = 1,
    replicates: int = 3,
    time_limit_s: float = 45.0,
    min_batches: int = 5,
    drop_cache: bool = True,
):
    """Measure one baseline kind at one thread count. Threads are limited by the
    caller via taskset; this just reports `threads` and sets num_workers=threads-1.

    drop_cache (FASTA only): evict each haplotype file's pages via posix_fadvise
    DONTNEED after every read, forcing cold storage reads. Default on — real
    bcftools-consensus cohorts use 2*n_samples FASTAs (tens of TB) that never
    fit in RAM. Pass --no-drop-cache to disable (for profiling / warm-cache runs)."""
    if kind not in ("fasta", "pybigwig"):
        raise ValueError(f"kind must be 'fasta' or 'pybigwig', got {kind!r}")
    seqlens = [2048, 16384, 131072, 1048576]
    run_kind(
        kind=kind, threads=threads, fasta=fasta, fasta2=fasta2,
        bigwig_table=bigwig_table if kind == "pybigwig" else None,
        bed_dir=bed_dir, grid_dir=grid_dir, seqlens=seqlens, results=results,
        n_samples=n_samples, mem_cap_bytes=int(mem_cap_gib * 2**30),
        burn_in=burn_in, replicates=replicates, time_limit_s=time_limit_s,
        min_batches=min_batches, drop_cache=drop_cache,
    )


if __name__ == "__main__":
    run(main)
