# GVL 0.26.0 throughput parity probe — design

_Date: 2026-06-03. Branch: `feat/gvl-026-parity-probe` (off `feat/variant-throughput-n-sweep`)._

## Background

The GVL manuscript throughput numbers (`results/{hap,track}_results.csv`) were collected on
**genvarloader 0.6.1**. A later refactor measured on **0.24.1** — with the same sweep cells —
ran **~18–20× slower** and hit **OOM** at larger seqlens. Root-cause analysis (the first half of
`../GenVarLoader/docs/superpowers/REGRESSIONS.md`) attributed this mainly to per-batch awkward-array
ragged assembly (`reverse_complement_ragged`, `to_padded`) on the getitem hot path, plus
throughput that *decreased* with more numba threads.

**0.26.0 is the direct fix attempt** (flat-buffer numba transforms, `_Flat`/`_FlatAnnotatedHaps`
reconstructors bypassing awkward). It also ships a new buffered dataloader
(`to_dataloader(mode="buffered", buffer_bytes=...)`) that coarsens fetching — the skill confirms
gvl throughput scales with fetch size, so one big fetch sliced into mini-batches beats many small
per-batch calls.

## Goal & success criterion

Determine whether **0.26.0 recovers ~parity with 0.6.1** for haplotype and track dataloading
throughput, and confirm the OOM is gone at the seqlens that previously blew up.

- **Success:** v0.26.0 throughput within a small factor of the v0.6.1 baseline on aligned cells
  (some slowdown acceptable given correctness/robustness improvements), no OOM across the 4 seqlens.
- **Diagnostic:** measure both `mode=None` (default) and `mode="buffered"` so we can attribute any
  recovery to the bare 0.26.0 refactor vs. the new buffered dataloader.

This is a **focused probe first** — a representative slice to confirm parity before committing to a
full multi-day grid replication. The full sweep (all datasets incl. UKBB, full thread×batch grid)
is a follow-up decision made *after* seeing probe results.

## Environment — `bench026`

New pixi feature/env, parallel to `bench061`:
- `genvarloader == 0.26.0`
- CPU torch (no GPU in the benchmark — async/double-buffered would give no benefit, so we use
  **single** buffered).

Default `bench` (0.24.1) and `bench061` (0.6.1) are left untouched. Add one bullet to the
"GenVarLoader version sensitivity" section of `CLAUDE.md` describing `bench026` and its purpose
(parity probe against the 0.6.1 manuscript numbers).

## Datasets — reuse-first

Reuse the already-built 0.24.x datasets; rebuild only if 0.26.0 cannot open them.

| Use | Path |
|---|---|
| 1KGP haps | `hap_track_throughput/datasets/1kgp/native/seqlen_{2048,16384,131072,1048576}.gvl` |
| TCGA haps + tracks | `hap_track_throughput/datasets/tcga-atac/native/seqlen_*.gvl` (has `genotypes/` + `intervals/read-depth/`) |

A one-line open-probe (`gvl.Dataset.open(path, fasta).with_len(L)[0:2]`) gates reuse. On a
format/version error, rebuild just that dataset with the existing `make_bed.py` (canonical chroms,
~100 samples) + a 0.26.0 `gvl.write`. FASTA paths come from `configs/{1kgp,tcga-atac}.config`.

## Reduced grid

All 4 seqlens; thinned thread×batch grid chosen to **align with cells that already exist in the
v0.6.1 baseline** (so the join is apples-to-apples), reusing the baseline's `npb ≤ 2**33` cap.

- **seqlens:** 2048, 16384, 131072, 1048576
- **threads:** {1, 16, 64} (single / mid / full)
- **batch_size:** 3 points per seqlen — small (bs=1), mid, and large near that seqlen's high end
  (bounded so `npb = seqlen × batch_size ≤ 2**33`)
- **dl modes:** `none`, `buffered`
- **datasets/modes:** 1KGP haps; TCGA haps; TCGA tracks
- ≈ 4 seqlens × 3 threads × 3 batches × 2 dl-modes × 3 (dataset,mode) combos × replicates —
  tractable in one interactive/SLURM session.

## Probe scripts — `hap_track_throughput/bin_gvl026/`

Mirrors the `bin_gvl061/` pattern (standalone, not wired into the heavy `benchmark.nf`).

**`benchmark_dl.py`** — single script:
- args: `--mode {haps,tracks}`, `--dl-mode {none,buffered}`, `--buffer-bytes` (default `2 * 2**30`
  = 2 GiB), plus results path / ds path / fasta / grid file (as in the existing bin/ scripts).
- Uses the forward-compatible fluent API: `gvl.Dataset.open(...).with_len(L)` with
  `.with_seqs("haplotypes")` + `.with_tracks(False)` for haps, or `.with_seqs(None)` +
  `.with_tracks("read-depth")` for tracks. `deterministic=True` (required by buffered haps; the
  default anyway).
- `to_dataloader(batch_size=bs, shuffle=False, mode=<none|"buffered">, buffer_bytes=2 GiB)`.
  `num_workers` left at 0 (buffered rejects >0).
- **Emits throughput directly in MiB/s**, computed exactly as `bin_gvl061` did
  (`n_elements × element_size / seconds / 2**20`; haps element_size 1, tracks 4), so the output
  CSV is directly comparable to the baseline. Schema:
  `dataset,backend,mode,threads,seqlen,batch_size,throughput (MiB/s)` (`backend=gvl026`).
- Burn-in + replicate timing loop as in the existing scripts.

**`probe026.sh`** — driver that loops (dataset, mode) × dl-mode, runs the open-probe/rebuild gate,
then the grid, writing idempotently into a new top-level `results_gvl026/` dir
(skip a cell whose CSV already exists, like `mem_driver.sh`).

## Analysis / output

Join `results_gvl026/` against `results/{hap,track}_results.csv` on
`(dataset, threads, seqlen, batch_size)` and report per-cell ratio `v0.26.0 / v0.6.1` for each
dl-mode — a table + a parity plot (scatter or grouped bar, x = baseline MiB/s, y = 0.26.0 MiB/s,
series = dl-mode, faceted by seqlen). Implemented either as a small standalone script under the
probe dir or as an addition to `scripts/plot.py`. This is the artifact that answers
"did 0.26.0 reach parity, and how much did buffering contribute?"

## Out of scope (for the probe)

- Full grid replication / UKBB / write & svar-convert benches.
- Memory (avg/peak RSS) regeneration — tracked separately in `HANDOFF.md`; can be added as a
  follow-up using `benchmark_dl.py --measure-memory` if the throughput probe looks good.
- `double_buffered` mode (no GPU consumer → no benefit).

## Decisions captured

- Scope: focused probe first.
- DL mode: both `none` + `buffered`.
- Env: new `bench026` env.
- Datasets: reuse existing if 0.26 opens them, else rebuild.
- Branch: off `feat/variant-throughput-n-sweep`.
- `buffer_bytes`: fixed 2 GiB.
- Standalone driver (not `benchmark.nf`); scripts emit MiB/s directly.
