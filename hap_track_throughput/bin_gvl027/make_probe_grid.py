#! /usr/bin/env python
"""Emit the reduced parity-probe grid (threads x batch_size x n_batches) for one seqlen.

Columns match make_launch_grid.py so benchmark_dl.py can read either:
    threads,batch_size,n_batches
"""

from pathlib import Path

import cyclopts

from _probe_common import PROBE_NPB_EXPS, batch_for_npb, n_batches_for

THREADS = (1, 16, 64)


def main(length: int, output: Path | None = None):
    import polars as pl

    rows = []
    for npb_exp in PROBE_NPB_EXPS:
        bs = batch_for_npb(length, npb_exp)
        nb = n_batches_for(2 ** npb_exp)
        for t in THREADS:
            rows.append({"threads": t, "batch_size": bs, "n_batches": nb})

    if output is None:
        output = Path.cwd() / f"probe_grid_{length}.csv"
    pl.from_dicts(rows).write_csv(output)
    print(f"WROTE {output} ({len(rows)} rows)")


if __name__ == "__main__":
    cyclopts.run(main)
