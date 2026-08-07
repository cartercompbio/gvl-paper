"""Guard: the private search/gather split must stay equivalent to the public fused read.

bench_svar2.py times _find_ranges (setup) and _gather_ranges (read) separately, mirroring
how GVL caches ranges at write time (GenVarLoader _write.py:1187). Those methods are
private, so this test fails loudly if a genoray upgrade removes or changes them rather
than letting the benchmark silently measure something else.
"""

import os

import numpy as np
import pytest

# genoray's SparseVar2 (and its private batch-query API) needs genoray>=3.3.1,
# which lives only in the isolated `svar2` pixi env -- the shared `bench`/
# default env stays pinned to genoray==2.9.0 (see pixi.toml). Gate on the pixi
# env itself, not on whether `SparseVar2` happens to be importable: keying off
# symbol presence would make a future rename of `SparseVar2` (which SHOULD
# fail this suite loudly, per this file's whole purpose) look like just
# another "wrong env" skip instead of a real break.
if os.environ.get("PIXI_ENVIRONMENT_NAME") != "svar2":
    pytest.skip(
        "SVAR2 lives only in the `svar2` pixi env -- run with "
        "`pixi run -e svar2 pytest variant_throughput/bin/tests/test_bench_svar2.py`",
        allow_module_level=True,
    )

from genoray import SparseVar2  # noqa: E402 -- must follow the env-gated skip above

from bench_svar2 import _svar2_n_calls, _svar2_search  # noqa: E402


def test_private_split_methods_exist(svar2_store):
    sv = SparseVar2(str(svar2_store))
    assert hasattr(sv, "_find_ranges"), "genoray removed SparseVar2._find_ranges"
    assert hasattr(sv, "_gather_ranges"), "genoray removed SparseVar2._gather_ranges"


@pytest.mark.parametrize(
    "subset_size",
    [None, 1, 2],
    ids=["no_subset", "one_sample", "two_samples"],
)
def test_split_matches_fused_read(svar2_store, contig, starts, ends, subset_size):
    """gather_ranges(find_ranges(...)) must equal the fused read_ranges.

    Parametrized over no subset, a 1-sample subset, and a 2-sample (= full
    cohort, but explicit) subset -- bench_svar2.py ALWAYS calls `_find_ranges`
    with `samples=unique_samples`, so the subset-aware path (`_sample_idxs`
    resolution, `sample_cols`, and the subset-aware indexing gather.rs does
    over the selected sample slots) is exactly what needs guarding, not just
    the whole-cohort default.
    """
    sv = SparseVar2(str(svar2_store))
    samples = None if subset_size is None else sv.available_samples[:subset_size]
    fused = sv.read_ranges(contig, starts, ends, samples=samples)
    bundle = sv._find_ranges(contig, starts, ends, samples=samples)
    split = sv._gather_ranges(contig, bundle)
    assert set(fused.keys()) == set(split.keys())
    for k in fused:
        np.testing.assert_array_equal(
            np.asarray(fused[k]), np.asarray(split[k]), err_msg=f"mismatch in {k!r}"
        )


def test_svar2_n_calls_indexes_by_sample_not_full_cohort(svar2_store, contig):
    """`_svar2_n_calls` must index region_counts's full-cohort sample axis down
    to the named sample, not sum the whole cohort.

    Region [0, 5) covers only the pos-3 SNP (0-based pos 2, see conftest's
    _SVAR2_VCF): S0 is het (1 carrier hap), S1 is hom-ref (0 carrier haps) --
    verified directly against `sv.region_counts` in review. A batch naming
    the zero-carrier sample must report 0, not the nonzero full-cohort total.
    """
    sv = SparseVar2(str(svar2_store))
    zero_carrier = [((contig, 0, 5), "S1")]
    assert _svar2_n_calls(sv, zero_carrier) == 0

    one_carrier = [((contig, 0, 5), "S0")]
    assert _svar2_n_calls(sv, one_carrier) == 1

    both = one_carrier + zero_carrier
    assert _svar2_n_calls(sv, both) == 1


def test_gather_covers_only_pair_cells_not_full_cross_product(svar2_store, contig):
    """Regression for the setup/read parity fix: the narrowed bundles
    `_svar2_search` hands to `_gather_ranges` must cover exactly the (region,
    sample) cells the pairs name, not the full region x unique-sample
    cross product `_find_ranges` computed.

    Two pairs, two different regions, two different samples -- so
    `unique_samples` for the batch has 2 entries, but each region only pairs
    with ONE of them. Before the fix, `_svar2_gather` replayed the single
    unnarrowed bundle spanning the full 2 regions x 2 samples cross product
    (4 cells) regardless of which cells the pairs actually named; at
    `query_length=2048` scale (~600 unique samples per batch) this was a
    ~600x inflation with `n_calls` still reporting only the true pair count.

    Both pairs are chosen at sites where the named sample is a HETEROZYGOUS
    carrier (exactly 1 carrier hap, verified against `sv.region_counts` in
    review), so the gathered (region, sample) cell count and `n_calls`
    coincide at 2 -- under the bug they would not (gathered cells = 4).
    """
    sv = SparseVar2(str(svar2_store))
    pairs = [
        ((contig, 0, 5), "S0"),  # pos 3 (0-based 2): S0 het -> 1 carrier hap
        ((contig, 10, 15), "S1"),  # pos 12 (0-based 11): S1 het -> 1 carrier hap
    ]

    narrowed_by_contig = _svar2_search(sv, pairs)
    gathered_cells = sum(
        nb["n_regions"] * nb["n_samples"]
        for _, narrowed in narrowed_by_contig
        for nb in narrowed
    )
    n_calls = _svar2_n_calls(sv, pairs)

    assert n_calls == 2, "fixture assumption: each named cell carries exactly 1 hap"
    assert gathered_cells == len(pairs)
    assert gathered_cells == n_calls
