"""Vibration anomaly detection on CWRU bearing data.

Models are trained on healthy windows only and asked to flag anything that does
not look healthy, which is how a deployed monitor meets a fault it has never seen.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import io as sio
from scipy import signal, stats

FS = 12_000  # target sample rate (Hz)

# CWRU file id -> (condition, motor load in HP). 12 kHz drive-end, 0.007" faults.
FILES = {
    97: ("normal", 0), 98: ("normal", 1), 99: ("normal", 2), 100: ("normal", 3),
    105: ("inner_race", 0), 106: ("inner_race", 1), 107: ("inner_race", 2), 108: ("inner_race", 3),
    118: ("ball", 0), 119: ("ball", 1), 120: ("ball", 2), 121: ("ball", 3),
    130: ("outer_race", 0), 131: ("outer_race", 1), 132: ("outer_race", 2), 133: ("outer_race", 3),
}


def load_de_signal(path: Path, native_fs: int = FS) -> np.ndarray:
    """Drive-end acceleration from a CWRU .mat file, resampled to FS if needed."""
    mat = sio.loadmat(path)
    key = next(k for k in mat if k.endswith("_DE_time"))
    x = mat[key].astype(np.float64).ravel()
    if native_fs != FS:
        g = np.gcd(native_fs, FS)
        x = signal.resample_poly(x, FS // g, native_fs // g)
    return x


def windows(x: np.ndarray, size: int = 2048, hop: int = 1024) -> np.ndarray:
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


N_BANDS = 32


def features(w: np.ndarray) -> np.ndarray:
    """Hand-crafted features per window: time-domain statistics + log band energies."""
    w = w - w.mean(axis=1, keepdims=True)
    rms = np.sqrt((w ** 2).mean(axis=1))
    peak = np.abs(w).max(axis=1)
    time_feats = np.stack([
        np.log(rms + 1e-12),
        np.log(peak + 1e-12),
        peak / (rms + 1e-12),          # crest factor
        stats.kurtosis(w, axis=1),
        stats.skew(w, axis=1),
    ], axis=1)
    spec = np.abs(np.fft.rfft(w * np.hanning(w.shape[1]), axis=1)) ** 2
    edges = np.linspace(0, spec.shape[1], N_BANDS + 1).astype(int)
    bands = np.stack([spec[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    return np.concatenate([time_feats, np.log(bands + 1e-12)], axis=1).astype(np.float32)


@dataclass
class Split:
    x_train: np.ndarray
    x_val: np.ndarray       # healthy, held out, used only to set the alarm threshold
    x_test: np.ndarray
    y_test: np.ndarray      # 0 healthy, 1 fault
    cond_test: np.ndarray   # condition name per test window
    load_test: np.ndarray
    x_val_target: np.ndarray | None = None  # short healthy clip at the unseen load (recalibration only)


def build_split(signals: dict[int, np.ndarray], test_loads: set[int] | None = None,
                train_frac: float = 0.5, val_frac: float = 0.2) -> Split:
    """Chronological split of each healthy recording; all fault windows go to test.

    With ``test_loads`` set, healthy data from those loads is excluded from training,
    which checks whether the monitor survives an operating point it never saw.
    """
    tr, va, vt, te, y, cond, load = [], [], [], [], [], [], []
    for fid, x in signals.items():
        c, hp = FILES[fid]
        f = features(windows(x))
        if c == "normal":
            n = len(f)
            a, b = int(n * train_frac), int(n * (train_frac + val_frac))
            if test_loads is None or hp not in test_loads:
                tr.append(f[:a]); va.append(f[a:b])
            else:
                vt.append(f[a:b])
            if test_loads is None or hp in test_loads:
                part = f[b:]  # same held-out tail either way, so the two settings compare fairly
                te.append(part); y += [0] * len(part); cond += [c] * len(part); load += [hp] * len(part)
        elif test_loads is None or hp in test_loads:
            te.append(f); y += [1] * len(f); cond += [c] * len(f); load += [hp] * len(f)
    return Split(np.concatenate(tr), np.concatenate(va), np.concatenate(te),
                 np.array(y), np.array(cond), np.array(load),
                 np.concatenate(vt) if vt else None)


def add_noise(x: np.ndarray, snr_db: float, ref_power: float, seed: int = 0) -> np.ndarray:
    """White noise at a fixed absolute level, set relative to the healthy machine's power.

    Using one level for every recording (rather than each recording's own power) mimics a
    loud factory floor: faulty bearings stay louder, healthy ones get buried.
    """
    rng = np.random.default_rng(seed)
    p = ref_power / 10 ** (snr_db / 10)
    return x + rng.standard_normal(len(x)) * np.sqrt(p)


class Standardizer:
    def fit(self, x):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        return self

    def __call__(self, x):
        return ((x - self.mu) / self.sd).astype(np.float32)
