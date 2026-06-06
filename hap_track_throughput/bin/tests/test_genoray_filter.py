import polars as pl
import pytest

from _genoray_filter import hap_safe_pl_filter, hap_safe_vcf_callable


@pytest.mark.parametrize(
    "alt,expected_keep",
    [
        (["A"], True),
        (["AT"], True),
        (["<DEL>"], False),
        (["G[chr2:321["], False),
        ([".TGCA"], False),
        (["TGCA."], False),
    ],
)
def test_pl_filter_keeps_only_expandable_alts(alt, expected_keep):
    # is_symbolic / is_breakend operate on the ALT list column.
    df = pl.DataFrame({"ALT": [alt]})
    kept = df.filter(hap_safe_pl_filter())
    assert (kept.height == 1) is expected_keep


@pytest.mark.parametrize(
    "alts,expected_keep",
    [
        (["A"], True),
        (["AT"], True),
        (["<DEL>"], False),
        (["G[chr2:321["], False),
        ([".TGCA"], False),
        (["TGCA."], False),
        (["A", "<INS>"], False),  # any disallowed ALT drops the record
    ],
)
def test_vcf_callable_matches_pl_filter(alts, expected_keep):
    keep = hap_safe_vcf_callable()
    assert keep(alts) is expected_keep


def test_flags_select_subset():
    # no_symbolic only: breakend ALT survives the pl_filter
    df = pl.DataFrame({"ALT": [["G[chr2:321["]]})
    assert df.filter(hap_safe_pl_filter(no_symbolic=True, no_breakend=False)).height == 1
    # no_breakend only: symbolic ALT survives
    df2 = pl.DataFrame({"ALT": [["<DEL>"]]})
    assert df2.filter(hap_safe_pl_filter(no_symbolic=False, no_breakend=True)).height == 1
    # neither flag: filter is a no-op (keeps everything)
    df3 = pl.DataFrame({"ALT": [["<DEL>"], ["G[chr2:321["]]})
    assert df3.filter(hap_safe_pl_filter(no_symbolic=False, no_breakend=False)).height == 2
