"""Hap-safe variant filter shared by the SVAR conversion + the VCF/PLINK bench.

genvarloader cannot expand symbolic (`<DEL>`) or breakend (`G[chr2:321[`) ALT
alleles into literal nucleotides, so both must be dropped before any variant
reaches haplotype buffers. genoray 2.9.0 ships the polars filter expressions
`genoray.exprs.is_symbolic` / `is_breakend` (public). PGEN readers take the
polars expr alone; VCF/BCF readers require BOTH a cyvcf2 callable and the
matching polars expr (genoray enforces the both-or-neither invariant).

Public-API only: combining genoray.exprs with `&`/`~` needs no `import polars`,
and the breakend regex is copied from genoray.exprs._BND_PATTERN's documented
form (kept in sync by the unit tests in tests/test_genoray_filter.py).
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

# Mirror of genoray.exprs._BND_PATTERN (VCF 4.x breakend ALT replacement string).
# Matches mate-pair forms (contain `[` or `]`) and single-breakend forms
# (a base adjacent to a `.`). A lone `.` (no-ALT) does not match.
_BND_PATTERN = r"[\[\]]|^\.[A-Za-z]|[A-Za-z]\.$"


def hap_safe_pl_filter(no_symbolic: bool = True, no_breakend: bool = True):
    """Polars filter expression keeping only haplotype-expandable variants.

    Returns a `pl.Expr`. With both flags False, returns an all-True no-op expr.
    """
    import genoray

    expr = None
    if no_symbolic:
        expr = ~genoray.exprs.is_symbolic
    if no_breakend:
        be = ~genoray.exprs.is_breakend
        expr = be if expr is None else (expr & be)
    if expr is None:
        import polars as pl

        return pl.lit(True)
    return expr


def hap_safe_vcf_callable(
    no_symbolic: bool = True, no_breakend: bool = True
) -> Callable[[Iterable[str]], bool]:
    """cyvcf2-style callable mirroring `hap_safe_pl_filter`.

    Accepts an iterable of ALT strings (a `cyvcf2.Variant.ALT`) and returns True
    to KEEP the record. Pass directly as `VCF(filter=...)` alongside
    `pl_filter=hap_safe_pl_filter(...)`.
    """

    def keep(alts: Iterable[str]) -> bool:
        alts = list(alts)
        if no_symbolic and any(a.startswith("<") for a in alts):
            return False
        if no_breakend and any(re.search(_BND_PATTERN, a) is not None for a in alts):
            return False
        return True

    return keep


def open_filtered_reader(variants, no_symbolic: bool = True, no_breakend: bool = True):
    """Open a genoray PGEN/VCF reader with the hap-safe filter applied.

    Dispatches on suffix: `.pgen` -> PGEN(filter=expr); `.bcf`/`.vcf`/`.vcf.gz`
    -> VCF(filter=callable, pl_filter=expr). Returns (reader, source_fmt).
    """
    import genoray

    name = str(variants).lower()
    pl_filter = hap_safe_pl_filter(no_symbolic, no_breakend)
    if name.endswith(".pgen"):
        return genoray.PGEN(variants, filter=pl_filter), "pgen"
    if name.endswith(".bcf") or name.endswith(".vcf") or name.endswith(".vcf.gz"):
        cb = hap_safe_vcf_callable(no_symbolic, no_breakend)
        source_fmt = "bcf" if name.endswith(".bcf") else "vcf"
        return genoray.VCF(variants, filter=cb, pl_filter=pl_filter), source_fmt
    raise ValueError(f"Unsupported variant format: {variants}")
