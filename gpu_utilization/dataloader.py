# %%
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import genvarloader as gvl
import numpy as np
import polars as pl
import pytorch_lightning as lit
import seqpro as sp
from einops import rearrange
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from typing_extensions import assert_never

with open("/carter/users/dlaub/data/TCGA/somatic/brca_wgs_atac/samples.txt", "r") as f:
    TCGA_BRCA_WGS_SAMPLES = [line.strip() for line in f]


# %%
class ATACDataModule(lit.LightningDataModule):
    def __init__(
        self,
        gvl_path: Union[str, Path],
        reference: Union[str, Path],
        fold_path: Union[str, Path],
        return_sequences: Literal["reference", "haplotypes"],
        return_tracks: Optional[Union[Literal[False], str, List[str]]] = None,
        counts: Optional[Union[str, Path, NDArray]] = None,
        seed: Optional[int] = None,
        rc_prob: Optional[float] = None,
        trimming: int = (2048 - 1000) // 2,
        dl_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """DataModule for ATAC-seq data.

        Notes
        -----
        `return_tracks` and `counts` are mutually exclusive. If `counts` is provided, `return_tracks` is ignored."""
        super().__init__()
        self.save_hyperparameters()

        self.gvl_path = gvl_path
        self.reference = reference
        self.seed = seed
        self.return_sequences = return_sequences
        self.rc_prob = rc_prob
        self.trimming = trimming
        self.dl_kwargs = dl_kwargs or {}

        with open(fold_path, "r") as f:
            fold_info = json.load(f)
        self.samples: Dict[str, List[str]] = fold_info["patients"]
        chrom_splits: Dict[str, List[str]] = fold_info["chroms"]

        chr_to_split = {}
        for split, chroms in chrom_splits.items():
            for chrom in chroms:
                chr_to_split[chrom] = split

        if counts is not None:
            # (regions samples)
            if isinstance(counts, (str, Path)):
                self.counts: Optional[NDArray[np.int32]] = np.load(counts)
            else:
                self.counts = counts
            jitter = 0
            return_tracks = False
        else:
            if return_tracks is None:
                raise ValueError("Either `return_tracks` or `counts` must be provided.")
            self.counts = counts
            jitter = None

        ds = gvl.Dataset.open(
            path=gvl_path,
            reference=reference,
            return_sequences=return_sequences,
            return_tracks=return_tracks,
            jitter=jitter,
            seed=seed,
        )

        self.samples["train"] = list(
            set(ds.samples) - set(self.samples["valid"]) - set(self.samples["test"])
        )

        self.regions = {
            k[0]: v
            for k, v in ds.get_bed()
            .with_columns(split=pl.col("chrom").replace(chr_to_split, default="train"))
            .partition_by("split", include_key=False, as_dict=True)
            .items()
        }

        if self.counts is not None:
            ds = ds.with_settings(return_indices=True)

            def transform(  # type: ignore
                haps: NDArray[np.bytes_],
                ds_idx: NDArray[np.intp],
                s_idx: NDArray[np.intp],
                r_idx: NDArray[np.intp],
            ):
                haps = haps[..., ds.max_jitter : -ds.max_jitter]
                haps = sp.DNA.ohe(haps).swapaxes(-2, -1).astype(np.float32)
                haps = haps.squeeze()
                counts = self.counts[r_idx, s_idx][:, None]  # type: ignore
                return {"haps": haps, "counts": counts}
        else:
            if self.return_sequences == "reference":

                def transform(ref: NDArray[np.bytes_], track: NDArray[np.float32]):  # type: ignore
                    _ref = sp.DNA.ohe(ref)
                    track = track[:, self.trimming : -self.trimming]
                    if rc_prob is not None and ds.rng.random() < rc_prob:
                        _ref = sp.DNA.reverse_complement(
                            _ref, length_axis=-1, ohe_axis=-2
                        )
                        track = track[..., ::-1].copy()
                    _ref = rearrange(_ref, "b l a -> b 1 a l").astype(np.float32)
                    track = rearrange(track, "b l -> b 1 l")
                    return {"haps": _ref, "track": track}

            elif self.return_sequences == "haplotypes":

                def transform(haps: NDArray[np.bytes_], tracks: NDArray[np.float32]):
                    # (b 2 l a)
                    _haps = sp.DNA.ohe(haps)
                    # (b 2 l) -> (b 1 l)
                    tracks = tracks[:, [0], self.trimming : -self.trimming]
                    if rc_prob is not None and ds.rng.random() < rc_prob:
                        _haps = sp.DNA.reverse_complement(
                            _haps, length_axis=-1, ohe_axis=-2
                        )
                        tracks = tracks[..., ::-1].copy()
                    # (b 2 a l)
                    _haps = _haps.swapaxes(-2, -1).astype(np.float32)
                    return {"haps": _haps, "track": tracks}
            else:
                assert_never(self.return_sequences)

        self.ds = ds.with_settings(transform=transform)
        self.train_ds = self.ds.subset_to(
            samples=self.samples["train"], regions=self.regions["train"]
        )
        self.val_ds = self.ds.subset_to(
            samples=self.samples["valid"], regions=self.regions["valid"]
        )
        self.test_ds = self.ds.subset_to(
            samples=self.samples["test"], regions=self.regions["test"]
        )

    def train_dataloader(self):
        return self.train_ds.to_dataloader(**self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return self.val_ds.to_dataloader(**self.dl_kwargs)

    def test_dataloader(self):
        return self.test_ds.to_dataloader(**self.dl_kwargs)


class BigWigFastaDataset(Dataset):
    def __init__(
        self,
        samples2paths: Union[str, Path, pl.DataFrame],
        bed: Union[str, Path, pl.DataFrame],
        trimming: int,
    ) -> None:
        self.trimming = trimming
        # sample,h1,h2,bw
        if isinstance(samples2paths, (str, Path)):
            samples2paths = pl.read_csv(samples2paths)

        self.fastas = {
            sample: (gvl.Fasta("h1", h1), gvl.Fasta("h2", h2))
            for sample, h1, h2, *_ in samples2paths.iter_rows()
        }
        self.bigwigs = gvl.BigWigs.from_table(
            "bw", samples2paths.select("sample", path="bw")
        )
        self.samples = list(self.fastas)
        self.n_samples = len(self.fastas)

        if isinstance(bed, (str, Path)):
            bed = gvl.read_bedlike(bed)
        self.bed = bed
        self.n_regions = len(self.bed)

    def __len__(self) -> int:
        return self.n_regions * self.n_samples

    @property
    def shape(self):
        return (self.n_regions, self.n_samples)

    def __getitem__(self, idx: int):
        region_idx, sample_idx = np.unravel_index(idx, self.shape)
        contig, start, end, *_ = self.bed.row(region_idx)
        sample = self.samples[sample_idx]
        h1, h2 = self.fastas[sample]
        # (p l)
        hap = np.stack(
            [h1.read(contig, start, end), h2.read(contig, start, end)], axis=0
        )
        # (p a l)
        hap = sp.DNA.ohe(hap)
        hap = rearrange(hap, "p l a -> p a l")
        # (1 l)
        track = self.bigwigs.read(contig, start, end, sample=sample)[
            :, self.trimming : -self.trimming
        ]
        return {"haps": hap.astype(np.float32), "track": track}


class FastaDataModule(lit.LightningDataModule):
    def __init__(
        self,
        samples2paths: Path,
        bed: Path,
        fold_path: Path,
        dl_kwargs: Optional[Dict[str, Any]] = None,
        trimming: int = (2048 - 1000) // 2,
        seqlen: int = 2048,
    ):
        self.trimming = trimming
        self.seqlen = seqlen

        with open(fold_path, "r") as f:
            fold_info = json.load(f)
        self.samples: Dict[str, List[str]] = fold_info["patients"]
        self.samples["train"] = list(
            set(TCGA_BRCA_WGS_SAMPLES)
            - set(self.samples["valid"])
            - set(self.samples["test"])
        )

        chrom_splits: Dict[str, List[str]] = fold_info["chroms"]
        chr_to_split = {}
        for split, chroms in chrom_splits.items():
            for chrom in chroms:
                chr_to_split[chrom] = split

        s2p = pl.read_csv(samples2paths)

        self.beds = {
            k[0]: v
            for k, v in gvl.with_length(gvl.read_bedlike(bed), self.seqlen)
            .with_columns(split=pl.col("chrom").replace(chr_to_split, default="train"))
            .partition_by("split", include_key=False, as_dict=True)
            .items()
        }
        self.train_ds = BigWigFastaDataset(
            s2p.filter(pl.col("sample").is_in(self.samples["train"])),
            self.beds["train"],
            trimming=self.trimming,
        )
        self.val_ds = BigWigFastaDataset(
            s2p.filter(pl.col("sample").is_in(self.samples["valid"])),
            self.beds["valid"],
            trimming=self.trimming,
        )
        self.test_ds = BigWigFastaDataset(
            s2p.filter(pl.col("sample").is_in(self.samples["test"])),
            self.beds["test"],
            trimming=self.trimming,
        )
        self.dl_kwargs = dl_kwargs or {}

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.dl_kwargs)
