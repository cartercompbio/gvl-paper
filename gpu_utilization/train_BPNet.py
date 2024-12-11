"""
#! NOTE !#
BPNet metrics will NOT work unless the ".squeeze()" calls are removed from the seqmodels.Module class
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from seqmodels import Module

sys.path.insert(0, "/cellar/users/dlaub/projects/tcga-atac")
from arch import BPNetHaps
from dataloader import ATACDataModule
from metrics import bpnetlite_loss, bpnetlite_metrics

# Set seed
np.random.seed(1234)

# Paths
fold = 0  # TODO: choose fold
data_dir = Path("/cellar/shared/carterlab/data/ml4gland/tcga-atac/data")
gvl_path = data_dir / "tcga-atac.gvl"
reference = data_dir / "GRCh38.d1.vd1.fa"
fold_path = data_dir / "splits" / f"fold_{fold}.json"
output_dir = f"/cellar/shared/carterlab/data/ml4gland/tcga-atac/models/variation_pilot/BPNet/HapsAndTracks/241016/gvl/fold_{fold}"
os.makedirs(output_dir, mode=0o777, exist_ok=True)
# os.chdir(output_dir)

# Report cuda availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dm = ATACDataModule(
    gvl_path=gvl_path,
    reference=reference,
    fold_path=fold_path,
    return_sequences="haplotypes",
    return_tracks="tn5",
    rc_prob=0.5,
    dl_kwargs=dict(
        batch_size=1024,
        pin_memory=True,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=8,
    ),
)
print(
    f"Train patients: {dm.train_ds.n_samples}, train regions: {dm.train_ds.n_regions}"
)
print(f"Valid patients: {dm.val_ds.n_samples}, valid regions: {dm.val_ds.n_regions}")
print(f"Test patients: {dm.test_ds.n_samples}, test regions: {dm.test_ds.n_regions}")

# Architecture
arch = BPNetHaps(trimming=(2048 - 1000) // 2).to(memory_format=torch.channels_last)
arch = torch.compile(arch)

# Create module for training
module = Module(
    arch=arch,
    input_vars=["haps"],
    output_vars=["profile", "counts"],
    target_vars=["track"],
    loss_fxn=bpnetlite_loss,
    val_metrics_fxn=bpnetlite_metrics,
    val_metrics_kwargs={"alpha": arch.alpha},
    optimizer_lr=1e-3,
)

logger = WandbLogger(
    name="GVL A30",
    project="gvl-paper",
    tags=["BPNet", "reference", "fold_0", "benchmark"],
    save_dir=Path(output_dir) / "logs",
)
trainer = Trainer(
    logger=False,
    max_epochs=10,
    val_check_interval=None,
    precision="bf16-mixed",
    benchmark=True,
    enable_progress_bar=True,
)

trainer.fit(module, train_dataloaders=dm.train_dataloader())

print("Done!")
