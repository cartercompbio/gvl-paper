"""Guard: the private search/gather split must stay equivalent to the public fused read.

bench_svar2.py times _find_ranges (setup) and _gather_ranges (read) separately, mirroring
how GVL caches ranges at write time (GenVarLoader _write.py:1187). Those methods are
private, so this test fails loudly if a genoray upgrade removes or changes them rather
than letting the benchmark silently measure something else.
"""

import numpy as np
import pytest

# genoray's SparseVar2 (and its private batch-query API) needs genoray>=3.4,
# which lives only in the isolated `svar2` pixi env -- the shared `bench`/
# default env stays pinned to genoray==2.9.0 (see pixi.toml). Skip cleanly
# rather than erroring at collection so `pixi run pytest variant_throughput/
# bin/tests/` still collects the rest of the suite under the default env; run
# this file for real with `pixi run -e svar2 pytest ...`.
genoray = pytest.importorskip("genoray")
SparseVar2 = getattr(genoray, "SparseVar2", None)
if SparseVar2 is None:
    pytest.skip(
        "genoray.SparseVar2 unavailable (genoray<3.4) -- run with "
        "`pixi run -e svar2 pytest variant_throughput/bin/tests/test_bench_svar2.py`",
        allow_module_level=True,
    )


def test_private_split_methods_exist(svar2_store):
    sv = SparseVar2(str(svar2_store))
    assert hasattr(sv, "_find_ranges"), "genoray removed SparseVar2._find_ranges"
    assert hasattr(sv, "_gather_ranges"), "genoray removed SparseVar2._gather_ranges"


def test_split_matches_fused_read(svar2_store, contig, starts, ends):
    """gather_ranges(find_ranges(...)) must equal the fused read_ranges."""
    sv = SparseVar2(str(svar2_store))
    fused = sv.read_ranges(contig, starts, ends)
    bundle = sv._find_ranges(contig, starts, ends)
    split = sv._gather_ranges(contig, bundle)
    assert set(fused.keys()) == set(split.keys())
    for k in fused:
        np.testing.assert_array_equal(
            np.asarray(fused[k]), np.asarray(split[k]), err_msg=f"mismatch in {k!r}"
        )
