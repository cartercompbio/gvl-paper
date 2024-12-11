from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from tqdm.auto import tqdm


class BPNetHaps(torch.nn.Module):
    def __init__(
        self,
        n_filters=64,
        n_layers=8,
        alpha=1,
        profile_output_bias=True,
        count_output_bias=True,
        name=None,
        trimming=None,
        verbose=True,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers

        self.alpha = alpha
        self.name = name or f"bpnet.{n_filters}.{n_layers}"
        self.trimming = trimming or 2**n_layers

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.irelu = torch.nn.ReLU()

        self.rconvs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    n_filters, n_filters, kernel_size=3, padding=2**i, dilation=2**i
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.rrelus = torch.nn.ModuleList(
            [torch.nn.ReLU() for _ in range(len(self.rconvs))]
        )

        self.fconv = torch.nn.Conv1d(
            n_filters, 1, kernel_size=75, padding=37, bias=profile_output_bias
        )

        self.linear = torch.nn.Linear(n_filters, 1, bias=count_output_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape = (batch, 2, 4, seqlen)

        Returns
        -------
        profile : torch.Tensor
        counts : torch.Tensor
        """
        p = x.shape[1]
        x = rearrange(x, "b p a l -> (b p) a l")

        start, end = self.trimming, x.shape[-1] - self.trimming

        x = self.irelu(self.iconv(x))
        for i in range(self.n_layers):
            x_conv = self.rrelus[i](self.rconvs[i](x))
            x = torch.add(x, x_conv)

        # profile prediction
        # (b p) f l -> (b p) 1 t
        y_profile = self.fconv(x)[..., start:end]
        # (b 1 l)
        y_profile = rearrange(y_profile, "(b p) 1 l -> b p 1 l", p=p).sum(dim=1)

        # counts prediction
        # (b p) f -> (b p) 1
        y_counts = self.linear(x[..., start - 37 : end + 37].mean(dim=-1))
        # (b 1)
        y_counts = rearrange(y_counts, "(b p) 1 -> b p 1", p=p).sum(dim=1)
        return y_profile, y_counts

    def predict(self, X, batch_size=64, verbose=False):
        with torch.no_grad():
            if not isinstance(X, tuple):
                starts = np.arange(0, X.shape[0], batch_size)
                X = tuple(X)
            else:
                starts = np.arange(0, X[0].shape[0], batch_size)
            ends = starts + batch_size

            y_profiles, y_counts = [], []
            for start, end in tqdm(zip(starts, ends), disable=not verbose):
                if not isinstance(X, tuple):
                    X_batch = X[start:end].cuda()
                else:
                    X_batch = tuple([X[i][start:end].cuda() for i in range(len(X))])

                y_profiles_, y_counts_ = self(X_batch)
                y_profiles_ = y_profiles_.cpu()
                y_counts_ = y_counts_.cpu()

                y_profiles.append(y_profiles_)
                y_counts.append(y_counts_)

            y_profiles = torch.cat(y_profiles)
            y_counts = torch.cat(y_counts)
            return y_profiles, y_counts
