#! /usr/bin/env python
import re
from enum import Enum
from pathlib import Path
from subprocess import CalledProcessError, run

import cyclopts
from attrs import define
from loguru import logger

proj_dir = Path(__file__).parent.parent
ds_dir = proj_dir / "data" / "datasets"
res_dir = proj_dir / "results"


@define(frozen=True)
class Dataset:
    fasta: Path
    variants: Path
    ds_dir: Path
    bigwig_table: Path | None = None


class DatasetEnum(Enum):
    TCGA_ATAC = Dataset(
        fasta=Path(
            "/cellar/users/dlaub/projects/tcga-atac/data/shared/GRCh38.d1.vd1.fa"
        ),
        variants=Path(
            "/cellar/users/dlaub/projects/tcga-atac/data/shared/merged.norm.bcf"
        ),
        ds_dir=ds_dir / "tcga-atac",
        bigwig_table=Path(
            "/carter/shared/data/ml4gland/tcga-atac/data/sample_to_bigwig.csv"
        ),
    )
    ONE_KGP = Dataset(
        fasta=Path(
            "/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa"
        ),
        variants=Path("/carter/users/dlaub/data/1kGP/1kGP.pgen"),
        ds_dir=ds_dir / "1kgp",
    )
    GDC = Dataset(
        fasta=Path(
            "/carter/shared/genomes/homo_sapiens/ensembl_grch38.p14_v108/Homo_sapiens.GRCh38.dna.toplevel.fa.bgz"
        ),
        variants=Path(
            "/carter/shared/data/gdc/somatic/wgs_DR45/results/gdc-wgs_DR45.svar"
        ),
        ds_dir=ds_dir / "gdc",
    )
    UKBB = Dataset(
        fasta=Path(
            "/carter/shared/genomes/homo_sapiens/ensembl_grch37.p13_v107/Homo_sapiens.GRCh37.dna.toplevel.fa.bgz"
        ),
        variants=Path(
            "/carter/shared/projects/InSNPtion/data/ukb/imputed_pgen/autosomes.aligned.name.pgen"
        ),
        ds_dir=ds_dir / "ukbb",
    )


def main(
    dataset: DatasetEnum,
    hap_results: Path | None = None,
    track_results: Path | None = None,
    min_npb: int | None = None,
    max_npb: int = 2**33,
    overwrite: bool = False,
    bench_haps: bool = True,
    bench_tracks: bool = False,
    redo_fails: Path | None = None,
):
    from itertools import product

    import numpy as np

    ds = dataset.value
    if ds.bigwig_table is None and bench_tracks:
        logger.warning("Ignoring bench-tracks since the dataset does not have them.")
        bench_tracks = False

    if hap_results is None:
        hap_results = (res_dir / "hap_results.csv").resolve()
    else:
        hap_results = hap_results.resolve()

    if track_results is None and ds.bigwig_table is not None:
        track_results = (res_dir / "track_results.csv").resolve()
    elif track_results is not None and ds.bigwig_table is not None:
        track_results = track_results.resolve()

    if dataset is dataset.UKBB:  # type: ignore
        just_chr22 = True
    else:
        just_chr22 = False
    seqlens = np.array([2048, 16384, 131072, 1048576])

    for length in seqlens:
        try:
            write(ds, length, overwrite=overwrite, just_chr22=just_chr22)
        except CalledProcessError as e:
            logger.error(f"Failed to launch write job: {e.stdout}, {e.stderr}")
            raise e

    if redo_fails is not None:
        with open(redo_fails) as f:
            groups = re.compile(r"(\w+)_([\w\-]+)_threads=(\d+)_seqlen=(\d+)_bs=(\d+)")

            @define
            class Experiment:
                type: str
                dataset: str
                threads: int
                seqlen: int
                batch_size: int

            to_redo = [
                Experiment(m[1], m[2].upper(), *map(int, m.groups()[2:]))
                for m in map(groups.match, f)
                if m is not None
            ]

            for exp in to_redo:
                _ds = DatasetEnum(exp.dataset).value
                npb = exp.seqlen * exp.batch_size
                n_batches = max(10, -(-(2**29) // npb))

                if exp.type == "haps":
                    launch_bench_haps(
                        hap_results,
                        _ds.ds_dir / f"seqlen={exp.seqlen}.gvl",
                        _ds.fasta,
                        exp.threads,
                        exp.batch_size,
                        n_batches,
                    )
                elif exp.type == "tracks":
                    assert track_results is not None
                    launch_bench_tracks(
                        track_results,
                        _ds.ds_dir / f"seqlen={exp.seqlen}.gvl",
                        _ds.fasta,
                        exp.threads,
                        exp.batch_size,
                        n_batches,
                    )

    if bench_haps or bench_tracks:
        logger.info("Launching benchmarks.")
        # 1 -> 64
        n_threads = 2 ** np.arange(0, 7, dtype=np.int64)
        for t, length in product(n_threads, seqlens):
            if min_npb is None or min_npb < length:
                min_npb = length
            min_batch_size = np.log2(min_npb // length)
            if min_batch_size < 1:
                min_batch_size = 1
            else:
                min_batch_size = min_batch_size.astype(np.int64)
            max_batch_size = np.log2(max_npb // length).astype(np.int64)
            batch_sizes = 2 ** np.arange(
                min_batch_size, max_batch_size + 1, dtype=np.int64
            )
            for bs in batch_sizes:
                ds_path = ds.ds_dir / f"seqlen={length}.gvl"
                npb = length * bs
                n_batches = max(10, -(-(2**29) // npb))

                if bench_haps:
                    try:
                        launch_bench_haps(
                            hap_results, ds_path, ds.fasta, t, bs, n_batches
                        )
                    except CalledProcessError as e:
                        print(e.stdout, e.stderr)
                        raise e

                if bench_tracks:
                    assert track_results is not None
                    try:
                        launch_bench_tracks(
                            track_results, ds_path, ds.fasta, t, bs, n_batches
                        )
                    except CalledProcessError as e:
                        print(e.stdout, e.stderr)
                        raise e


def write(
    dataset: Dataset, length: int, overwrite: bool = False, just_chr22: bool = False
):
    ds_path = dataset.ds_dir / f"seqlen={length}.gvl"
    if dataset == DatasetEnum.UKBB.value and length == 1048576 or length == 2048:
        mem = "256G"
        max_mem = "64G"
    else:
        mem = "64G"
        max_mem = "4G"
    if not ds_path.exists() or overwrite:
        logger.info(f"Writing dataset {ds_path}.")
        bed_fname = f"tile_{length}" + ("_chr22" if just_chr22 else "") + ".bed"
        bed = proj_dir / "beds" / bed_fname
        cmd = [
            "sbatch",
            f"--output={res_dir / 'out' / f'{ds_path.parent.name}_seqlen={length}.out'}",
            f"--error={res_dir / 'err' / f'{ds_path.parent.name}_seqlen={length}.err'}",
            "--partition=carter-compute",
            "--cpus-per-task=16",
            f"--mem={mem}",
            "--time=7-00",
            "genvarloader",
            str(ds_path),
            str(bed),
            f"--variants={dataset.variants}",
            f"--length={length}",
            "--overwrite",
            f"--max-memory={max_mem}",
            "--log-level=DEBUG",
        ]
        if dataset.bigwig_table is not None:
            cmd.append(f"--bigwig-table={dataset.bigwig_table}")
        run(cmd, check=True)


def launch_bench_haps(
    results: Path,
    ds_path: Path,
    fasta: Path,
    n_threads: int,
    batch_size: int,
    n_batches: int,
):
    out_file = f"haps_{ds_path.parent.name}_threads={n_threads}_{ds_path.stem}_bs={batch_size}.out"
    err_file = f"haps_{ds_path.parent.name}_threads={n_threads}_{ds_path.stem}_bs={batch_size}.err"
    cmd = [
        "sbatch",
        "--partition=carter-compute",
        f"--cpus-per-task={n_threads}",
        f"--output={res_dir / 'out' / out_file}",
        f"--error={res_dir / 'err' / err_file}",
        "--nodelist=carter-cn-04",
        "--mem=16G",
        str(proj_dir / "benchmark_haps.py"),
        str(results),
        str(ds_path),
        str(fasta),
        str(batch_size),
        str(n_batches),
    ]
    run(cmd, check=True)


def launch_bench_tracks(
    results: Path,
    ds_path: Path,
    fasta: Path,
    n_threads: int,
    batch_size: int,
    n_batches: int,
):
    out_file = f"tracks_{ds_path.parent.name}_threads={n_threads}_{ds_path.stem}_bs={batch_size}.out"
    err_file = f"tracks_{ds_path.parent.name}_threads={n_threads}_{ds_path.stem}_bs={batch_size}.err"
    cmd = [
        "sbatch",
        "--partition=carter-compute",
        f"--cpus-per-task={n_threads}",
        f"--output={res_dir / 'out' / out_file}",
        f"--error={res_dir / 'err' / err_file}",
        "--nodelist=carter-cn-04",
        "--mem=16G",
        str(proj_dir / "benchmark_tracks.py"),
        str(results),
        str(ds_path),
        str(fasta),
        str(batch_size),
        str(n_batches),
    ]
    run(cmd, check=True)


if __name__ == "__main__":
    cyclopts.run(main)
