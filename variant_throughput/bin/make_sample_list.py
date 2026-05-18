#! /usr/bin/env python

import random
import sys
from pathlib import Path

from cyclopts import run


def main(
    svar: Path,
    output: Path,
    n: int = 0,
    seed: int = 0,
    print_total: bool = False,
):
    """Emit a deterministic subset of samples from a SparseVar dataset.

    With --print-total, prints the total sample count to stdout and exits
    without writing the output file.

    Otherwise: shuffles _svar.available_samples with `seed`, takes the first
    `n` IDs, and writes them one-per-line to `output`. If `n` is 0 or >= total,
    writes every sample (full cohort passthrough).
    """
    from genoray import SparseVar

    _svar = SparseVar(svar)
    available = list(_svar.available_samples)

    if print_total:
        sys.stdout.write(f"{len(available)}\n")
        return

    if n <= 0 or n >= len(available):
        chosen = available
    else:
        rng = random.Random(seed)
        shuffled = available.copy()
        rng.shuffle(shuffled)
        chosen = shuffled[:n]

    output.write_text("\n".join(chosen) + "\n")


if __name__ == "__main__":
    run(main)
