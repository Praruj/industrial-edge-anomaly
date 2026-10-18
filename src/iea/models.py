"""Small models that fit on an edge device."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TinyAE(nn.Module):
    """Dense autoencoder on standardized vibration features (~4.5k parameters)."""

    def __init__(self, d_in: int, hidden: int = 48, bottleneck: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class AEScorer(nn.Module):
    """Wraps the AE so the exported graph outputs the anomaly score directly."""

    def __init__(self, ae: TinyAE):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        return ((self.ae(x) - x) ** 2).mean(dim=1)


def train_ae(x: np.ndarray, epochs: int = 200, seed: int = 0, lr: float = 3e-3) -> TinyAE:
    torch.manual_seed(seed)
    ae = TinyAE(x.shape[1])
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-5)
    xt = torch.from_numpy(x)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            b = xt[perm[i:i + 128]]
            loss = ((ae(b) - b) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return ae.eval()


@torch.no_grad()
def ae_scores(ae: TinyAE, x: np.ndarray) -> np.ndarray:
    return AEScorer(ae)(torch.from_numpy(x)).numpy()
