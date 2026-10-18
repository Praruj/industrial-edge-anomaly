"""Visual inspection on VisA with a compact PatchCore.

PatchCore needs no training: it stores locally-aware patch features from images of
good parts (a "memory bank") and scores a new image by how far its patches are from
anything seen before. The memory bank is shrunk with greedy coreset selection so the
whole detector fits comfortably on an edge device.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

IMG = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------- data ----------
@dataclass
class Sample:
    image: Path
    label: int          # 0 good, 1 defect
    mask: Path | None


def visa_split(root: Path, category: str) -> tuple[list[Sample], list[Sample]]:
    """Official VisA 1-class split (split_csv/1cls.csv)."""
    train, test = [], []
    with open(root / "split_csv" / "1cls.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["object"] != category:
                continue
            s = Sample(root / row["image"], int(row["label"] == "anomaly"),
                       root / row["mask"] if row.get("mask") else None)
            (train if row["split"] == "train" else test).append(s)
    return train, test


def load_image(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


def load_mask(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((IMG, IMG), dtype=bool)
    return np.asarray(Image.open(path).convert("L").resize((IMG, IMG), Image.NEAREST)) > 0


# ---------- features ----------
class PatchFeatures(nn.Module):
    """ResNet layer2 + layer3 features, 3x3 neighbourhood-averaged, on a 32x32 grid."""

    def __init__(self, backbone: nn.Module | None = None):
        super().__init__()
        if backbone is None:
            from torchvision.models import ResNet18_Weights, resnet18
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        b = backbone
        self.stem = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool, b.layer1)
        self.layer2, self.layer3 = b.layer2, b.layer3
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f3 = F.interpolate(self.pool(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([self.pool(f2), f3], dim=1)  # (B, 384, 32, 32)


@torch.no_grad()
def extract(model: nn.Module, images: list[np.ndarray], batch: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch):
        out.append(model(torch.from_numpy(np.stack(images[i:i + batch]))).numpy())
    return np.concatenate(out)


def to_patches(feats: np.ndarray) -> np.ndarray:
    b, c, h, w = feats.shape
    return feats.transpose(0, 2, 3, 1).reshape(b, h * w, c)


# ---------- memory bank ----------
def greedy_coreset(x: np.ndarray, n: int, proj_dim: int = 128, seed: int = 0) -> np.ndarray:
    """Indices of a k-center coreset (Sener & Savarese), computed in a random projection."""
    rng = np.random.default_rng(seed)
    n = min(n, len(x))
    xt = torch.from_numpy(x)
    if x.shape[1] > proj_dim:
        p = torch.from_numpy(rng.standard_normal((x.shape[1], proj_dim)).astype(np.float32))
        xt = xt @ p
    idx = [int(rng.integers(len(x)))]
    dist = torch.cdist(xt, xt[idx]).squeeze(1)
    for _ in range(n - 1):
        j = int(torch.argmax(dist))
        idx.append(j)
        dist = torch.minimum(dist, torch.cdist(xt, xt[j:j + 1]).squeeze(1))
    return np.array(idx)


def coreset_order(train_patches: np.ndarray, per_image: int = 256, bank_size: int = 10_000,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Patch pool and greedy coreset order. Greedy k-center is nested: the first k picks
    are the k-sized coreset, so one run gives every smaller memory bank for free."""
    rng = np.random.default_rng(seed)
    b, p, c = train_patches.shape
    pick = np.stack([rng.choice(p, size=min(per_image, p), replace=False) for _ in range(b)])
    pool = np.take_along_axis(train_patches, pick[..., None], axis=1).reshape(-1, c)
    return pool, greedy_coreset(pool, bank_size, seed=seed)


def build_memory_bank(train_patches: np.ndarray, per_image: int = 256, bank_size: int = 10_000,
                      seed: int = 0) -> np.ndarray:
    pool, order = coreset_order(train_patches, per_image, bank_size, seed)
    return pool[order]


def patch_scores(patches: np.ndarray, bank: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Nearest-neighbour distance of every patch to the memory bank. patches: (B, P, C)."""
    bt = torch.from_numpy(bank)
    flat = torch.from_numpy(patches.reshape(-1, patches.shape[-1]))
    d = torch.cat([torch.cdist(flat[i:i + chunk], bt).min(dim=1).values for i in range(0, len(flat), chunk)])
    return d.numpy().reshape(patches.shape[:2])


def anomaly_maps(scores: np.ndarray, grid: int = 32, sigma: float = 4.0) -> np.ndarray:
    """Patch scores -> smoothed full-resolution heatmaps (B, IMG, IMG)."""
    m = torch.from_numpy(scores.reshape(-1, 1, grid, grid))
    m = F.interpolate(m, size=(IMG, IMG), mode="bilinear", align_corners=False)
    k = int(4 * sigma + 1) | 1
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    m = F.conv2d(F.pad(m, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k))
    m = F.conv2d(m, g.view(1, 1, k, 1))
    return m[:, 0].numpy()
