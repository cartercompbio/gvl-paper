"""Pure helpers shared by the GVL 0.26.0 parity-probe scripts.

No genvarloader / torch imports here so the math stays unit-testable under any env.
"""

from __future__ import annotations

# Small / mid / large fetch sizes, as log2(nucleotides-per-batch). Each value is
# present in the v0.6.1 baseline grid for every probe seqlen, so the parity join matches.
PROBE_NPB_EXPS: tuple[int, ...] = (21, 25, 29)


def batch_for_npb(seqlen: int, npb_exp: int) -> int:
    """batch_size such that seqlen * batch_size == 2**npb_exp.

    Probe seqlens are powers of two, so this is exact. Raises if 2**npb_exp < seqlen
    (would imply batch_size < 1).
    """
    npb = 2 ** npb_exp
    if npb < seqlen:
        raise ValueError(f"npb 2**{npb_exp}={npb} < seqlen {seqlen}: batch_size would be < 1")
    bs, rem = divmod(npb, seqlen)
    if rem != 0:
        raise ValueError(f"seqlen {seqlen} does not divide npb 2**{npb_exp}")
    return bs


def n_batches_for(npb: int) -> int:
    """Number of measured batches for a cell: clip(2**29 // npb, 10, 200)."""
    raw = (2 ** 29) // max(1, npb)
    return min(200, max(10, raw))


def n_bytes(batch) -> int:
    """Total bytes in a batch, supporting numpy arrays and torch-like tensors."""
    if hasattr(batch, "itemsize") and hasattr(batch, "size"):  # numpy ndarray
        return int(batch.size) * int(batch.itemsize)
    return int(batch.numel()) * int(batch.element_size())  # torch.Tensor


def mib_per_s(total_bytes: int, seconds: float) -> float:
    """Throughput in MiB/s. Matches bin_gvl061's convention."""
    return total_bytes / seconds / 2 ** 20
