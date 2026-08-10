"""Guard: the private read-bound search/gather split must stay equivalent to the public fused read.

bench_svar2.py times `_find_haps_ranges` + `HapRangesRect.select` (setup) and
`_gather_haps_readbound` (read) separately, mirroring how GVL caches ranges at
write time (GenVarLoader _write.py:1187). Those methods are private, so this
test fails loudly if a genoray upgrade removes or changes them rather than
letting the benchmark silently measure something else.
"""

import os

import numpy as np
import pytest

# genoray's SparseVar2 (and its private batch-query API) lives only in the
# isolated `svar2` pixi env -- the shared `bench`/default env stays pinned to
# genoray==2.9.0 (see pixi.toml). Gate on the pixi env itself, not on whether
# `SparseVar2` happens to be importable: keying off symbol presence would make a
# future rename of `SparseVar2` (which SHOULD fail this suite loudly, per this
# file's whole purpose) look like just another "wrong env" skip instead of a
# real break.
if os.environ.get("PIXI_ENVIRONMENT_NAME") != "svar2":
    pytest.skip(
        "SVAR2 lives only in the `svar2` pixi env -- run with "
        "`pixi run -e svar2 pytest variant_throughput/bin/tests/test_bench_svar2.py`",
        allow_module_level=True,
    )

from genoray import SparseVar2  # noqa: E402 -- must follow the env-gated skip above

from bench_svar2 import _svar2_gather, _svar2_n_calls, _svar2_search  # noqa: E402


def _bits(packed, bit0: int, n: int):
    """`n` LSB-first presence bits starting at absolute bit offset `bit0`."""
    if n == 0:
        return np.zeros(0, bool)
    lo, hi = bit0 // 8, (bit0 + n + 7) // 8
    flat = np.unpackbits(np.asarray(packed, np.uint8)[lo:hi], bitorder="little")
    off = bit0 - lo * 8
    return flat[off : off + n].astype(bool)


def _fused_hap(br, r: int, s: int, p: int) -> set:
    """`(position, key)` set for one hap of a fused `read_ranges` result."""
    n_s, ploidy = int(br["n_samples"]), int(br["ploidy"])
    h = (r * n_s + s) * ploidy + p
    off = np.asarray(br["vk_off"])
    out = set(
        zip(
            np.asarray(br["vk_pos"])[off[h] : off[h + 1]].tolist(),
            np.asarray(br["vk_key"])[off[h] : off[h + 1]].tolist(),
        )
    )
    ds, de = np.asarray(br["dense_range"])[r]
    present = _bits(
        br["dense_present"], int(np.asarray(br["dense_present_off"])[h]), de - ds
    )
    out |= set(
        zip(
            np.asarray(br["dense_pos"])[ds:de][present].tolist(),
            np.asarray(br["dense_key"])[ds:de][present].tolist(),
        )
    )
    return out


def _split_hap(br, q: int, p: int) -> set:
    """`(position, key)` set for one hap of a read-bound `BatchResultSplit`.

    The split result keeps the two dense classes separate (no contig-wide
    union is ever built), so reconstructing a hap means merging var_key with
    BOTH -- this is what a consumer like gvl does downstream.
    """
    ploidy = int(br["ploidy"])
    h = q * ploidy + p
    off = np.asarray(br["vk_off"])
    out = set(
        zip(
            np.asarray(br["vk_pos"])[off[h] : off[h + 1]].tolist(),
            np.asarray(br["vk_key"])[off[h] : off[h + 1]].tolist(),
        )
    )
    for cls in ("snp", "indel"):
        ds, de = np.asarray(br[f"dense_{cls}_range"])[q]
        present = _bits(
            br[f"dense_{cls}_present"],
            int(np.asarray(br[f"dense_{cls}_present_off"])[h]),
            de - ds,
        )
        out |= set(
            zip(
                np.asarray(br[f"dense_{cls}_pos"])[ds:de][present].tolist(),
                np.asarray(br[f"dense_{cls}_key"])[ds:de][present].tolist(),
            )
        )
    return out


def test_private_split_methods_exist(svar2_store):
    sv = SparseVar2(str(svar2_store))
    assert hasattr(sv, "_find_haps_ranges"), (
        "genoray removed SparseVar2._find_haps_ranges"
    )
    assert hasattr(sv, "_gather_haps_readbound"), (
        "genoray removed SparseVar2._gather_haps_readbound"
    )


@pytest.mark.parametrize(
    "subset",
    [None, ["S1"], ["S1", "S0"]],
    ids=["no_subset", "second_sample_only", "reordered_full"],
)
def test_split_matches_fused_read(svar2_store, contig, starts, ends, subset):
    """`_gather_haps_readbound(select(_find_haps_ranges(...)))` must equal the
    fused `read_ranges` over the same (region, sample) cells.

    The subsets are deliberately NON-identity. The fixture has exactly two
    samples, so `available_samples[:1]`/`[:2]` would be identity prefixes and a
    gather that ignored `sample_cols` entirely would pass both. `["S1"]` puts
    original sample 1 at slot 0, and `["S1", "S0"]` reverses the cohort -- both
    fail loudly if the selected-sample indirection is dropped.
    """
    sv = SparseVar2(str(svar2_store))
    names = sv.available_samples if subset is None else subset
    rect = sv._find_haps_ranges(contig, starts, ends, samples=subset)

    # Every (region, slot) cell of the rectangle, as a flat pair list.
    n_r, n_s = len(starts), len(names)
    r_idx = np.repeat(np.arange(n_r), n_s)
    s_idx = np.tile(np.arange(n_s), n_r)
    split = sv._gather_haps_readbound(contig, rect.select(r_idx, s_idx))

    fused = sv.read_ranges(contig, starts, ends, samples=subset)
    for q, (r, s) in enumerate(zip(r_idx, s_idx)):
        for p in range(sv.ploidy):
            assert _split_hap(split, q, p) == _fused_hap(fused, int(r), int(s), p), (
                f"region {r}, sample {names[s]}, ploid {p}"
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
    """The timed read must cover exactly the (region, sample) cells the pairs
    name -- one gather call per (batch, contig), no cross-product.

    Two pairs, two different regions, two different samples: the cross-product
    is 4 cells and the pairs name 2. `vk_off` has one entry per gathered hap,
    so its length is a DIRECT measurement of how many cells the gather touched
    (not an assertion about the code path). Under the old rectangle-shaped
    `_gather_ranges` bundle this was either 4 cells in one call, or 2 cells
    across 2 calls each rebuilding the contig-wide dense union; at
    query_length=2048 scale (341 unique samples per batch) that was a 16-17x
    per-call floor inflation with `n_calls` still reporting only the pair count.

    Both pairs sit at sites where the named sample is a HETEROZYGOUS carrier
    (exactly 1 carrier hap, verified against `sv.region_counts` in review), so
    the gathered-cell count and `n_calls` coincide at 2.
    """
    sv = SparseVar2(str(svar2_store))
    pairs = [
        ((contig, 0, 5), "S0"),  # pos 3 (0-based 2): S0 het -> 1 carrier hap
        ((contig, 10, 15), "S1"),  # pos 12 (0-based 11): S1 het -> 1 carrier hap
    ]

    hap_ranges_by_contig = _svar2_search(sv, pairs)
    assert len(hap_ranges_by_contig) == 1, "single contig -> a single gather call"

    gathered_cells = 0
    for c, hr in hap_ranges_by_contig:
        split = sv._gather_haps_readbound(c, hr)
        gathered_cells += (len(np.asarray(split["vk_off"])) - 1) // int(
            split["ploidy"]
        )

    n_calls = _svar2_n_calls(sv, pairs)
    assert n_calls == 2, "fixture assumption: each named cell carries exactly 1 hap"
    assert gathered_cells == len(pairs)
    assert gathered_cells == n_calls


def test_gather_replays_without_further_search(svar2_store, contig):
    """`_svar2_gather` must be callable repeatedly on the cached ranges -- the
    read loop (`run_stream`/`drive_loop`) replays the same payload many times.
    """
    sv = SparseVar2(str(svar2_store))
    pairs = [((contig, 0, 20), "S0"), ((contig, 5, 30), "S1")]
    cached = _svar2_search(sv, pairs)
    for _ in range(3):
        _svar2_gather(sv, cached)
