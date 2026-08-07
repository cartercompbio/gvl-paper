"""Put the parent `bin/` dir on sys.path so tests import sibling modules by bare
name (`_streaming`, `_pairs`) exactly as Nextflow does on PATH."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 40 bp reference; the REF bases below match this exactly (1-based VCF POS):
# POS 3 = 'A', POS 7 = 'C', POS 12..14 = 'GTA'. Mirrors genoray's own
# tests/conftest.py::svar2_store fixture content, but built via the public
# `SparseVar2.from_vcf` classmethod (per task-4-brief.md Step 1) rather than
# genoray's internal `_core.run_conversion_pipeline`.
_SVAR2_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"

_SVAR2_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


@pytest.fixture(scope="session")
def svar2_store(tmp_path_factory) -> Path:
    """A tiny SVAR2 store (2 samples, 1 contig, 3 variants) for the bench_svar2 guard test."""
    from genoray import SparseVar2

    d = tmp_path_factory.mktemp("svar2")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR2_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR2_VCF)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(vcf)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)

    out = d / "store"
    SparseVar2.from_vcf(out, gz, ref, threads=1)
    assert (out / "meta.json").exists(), "SVAR2 conversion did not finish"
    return out


@pytest.fixture
def contig() -> str:
    return "chr1"


@pytest.fixture
def starts() -> list[int]:
    return [0, 5]


@pytest.fixture
def ends() -> list[int]:
    return [40, 20]
