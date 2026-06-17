import sys
from pathlib import Path

import polars as pl

BIN = Path(__file__).resolve().parents[1]
ROOT = BIN.parents[1]
sys.path.insert(0, str(BIN))


def _gvl_cells(glob):
    frames = [pl.read_csv(p) for p in (ROOT / "results_gvl027").glob(glob)]
    df = pl.concat(frames)
    return set(
        (int(t), int(s), int(b))
        for t, s, b in df.select("threads", "seqlen", "batch_size").unique().iter_rows()
    )


def test_generated_grid_matches_gvl_haps_cells():
    """make_launch_grid.py defaults must reproduce the exact (threads,seqlen,batch) cells the 0.27 haps grid ran."""
    grid_cells = set()
    for seqlen in (2048, 16384, 131072, 1048576):
        g = pl.read_csv(BIN / "beds" / f"grid_{seqlen}.csv")
        for t, b in g.select("threads", "batch_size").unique().iter_rows():
            grid_cells.add((int(t), int(seqlen), int(b)))
    assert grid_cells == _gvl_cells("haps/*_none.csv")
