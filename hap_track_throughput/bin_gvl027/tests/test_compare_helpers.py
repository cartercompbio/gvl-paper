import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from compare_to_baseline import norm_dataset, parse_mode_dir


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1KGP", "1kgp"),
        ("TCGA_ATAC", "tcga-atac"),
        ("UKBB", "ukbb"),
        ("1kgp", "1kgp"),
        ("tcga-atac", "tcga-atac"),
    ],
)
def test_norm_dataset(raw, expected):
    assert norm_dataset(raw) == expected


@pytest.mark.parametrize(
    "dirname,expected",
    [
        ("haps", ("haps", False)),
        ("tracks", ("tracks", False)),
        ("haps_memory", ("haps", True)),
        ("tracks_memory", ("tracks", True)),
    ],
)
def test_parse_mode_dir(dirname, expected):
    assert parse_mode_dir(dirname) == expected


def test_parse_mode_dir_rejects_unknown():
    with pytest.raises(ValueError):
        parse_mode_dir("svar_convert")
