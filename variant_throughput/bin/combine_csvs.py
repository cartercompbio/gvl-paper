#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def bench(output: Path, *inputs: Path):
    import polars as pl

    pl.concat([pl.read_csv(f) for f in inputs]).sort(
        ["query_length", "replicate"], maintain_order=True
    ).write_csv(output)


if __name__ == "__main__":
    run(bench)
