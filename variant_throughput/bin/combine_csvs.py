#! /usr/bin/env python

from pathlib import Path

from cyclopts import run


def bench(output: Path, *inputs: Path):
    import polars as pl

    # `setup_ns` is empty for the methods with no cached-setup phase (bcf,
    # pgen) and populated for the ones that have one (svar, svar2,
    # presubset_bcf). read_csv infers an all-null column as String, so a
    # CROSS-METHOD concat mixes String and Int64 and dies with
    # `SchemaError: type String is incompatible with expected type Int64`.
    # Force the dtype wherever the column exists. The within-method combines
    # this same script performs inside the pipeline (COMBINE_THROUGHPUT /
    # COMBINE_MEMORY) are unaffected -- their inputs already agree -- and the
    # cast is meaning-preserving: an empty setup_ns stays null, it does not
    # become 0, so "no setup phase" is never silently read as "setup was
    # instantaneous".
    frames = []
    for f in inputs:
        df = pl.read_csv(f)
        if "setup_ns" in df.columns:
            df = df.with_columns(pl.col("setup_ns").cast(pl.Int64))
        frames.append(df)

    # Sort keys beyond (query_length, replicate) are tiebreakers only, for
    # DETERMINISM: rows that tie on those two -- every cohort-sweep row shares
    # query_length=131072 and replicate -- were previously left in whatever
    # order Nextflow's channel happened to emit the shards, so re-merging the
    # same unchanged inputs produced a spurious whole-block diff in the
    # committed CSV (observed: 60 reordered lines in memory.csv from a re-run
    # that did not touch a single memory task). Content is unaffected.
    keys = ["query_length", "replicate"]
    keys += [c for c in ("method", "n_samples") if c in frames[0].columns]

    pl.concat(frames).sort(keys, maintain_order=True).write_csv(output)


if __name__ == "__main__":
    run(bench)
