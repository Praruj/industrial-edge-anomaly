"""Fast tests on synthetic data: no downloads, no pretrained weights."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iea import edge, vision  # noqa: E402
from iea.models import AEScorer, ae_scores, train_ae  # noqa: E402
from iea.vibration import FILES, N_BANDS, Standardizer, build_split, features, windows  # noqa: E402

FS = 12_000


def healthy(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * 30 * t) + 0.3 * rng.standard_normal(n)


def faulty(n, seed):
    x = healthy(n, seed)
    x[:: FS // 100] += 6.0  # periodic impacts, like a spall on a race
    return x


def synthetic_signals(n=60_000):
    return {fid: (healthy if c == "normal" else faulty)(n, fid) for fid, (c, _) in FILES.items()}


def test_windows_shape_and_overlap():
    w = windows(np.arange(10_000, dtype=float), size=2048, hop=1024)
    assert w.shape == (8, 2048)
    assert w[1, 0] == 1024


def test_features_finite_and_sized():
    f = features(windows(healthy(20_000, 0)))
    assert f.shape[1] == 5 + N_BANDS
    assert np.isfinite(f).all()


def test_impulses_raise_kurtosis():
    kh = features(windows(healthy(20_000, 0)))[:, 3].mean()
    kf = features(windows(faulty(20_000, 0)))[:, 3].mean()
    assert kf > kh + 3


def test_split_trains_on_healthy_only():
    s = build_split(synthetic_signals())
    assert set(np.unique(s.cond_test)) == {"normal", "inner_race", "ball", "outer_race"}
    assert (s.y_test == (s.cond_test != "normal")).all()


def test_unseen_load_split_excludes_that_load_from_training():
    sig = synthetic_signals()
    full, held = build_split(sig), build_split(sig, test_loads={3})
    assert len(held.x_train) < len(full.x_train)
    assert set(np.unique(held.load_test)) == {3}


def test_autoencoder_separates_synthetic_faults():
    s = build_split(synthetic_signals())
    std = Standardizer().fit(s.x_train)
    ae = train_ae(std(s.x_train), epochs=60)
    assert roc_auc_score(s.y_test, ae_scores(ae, std(s.x_test))) > 0.95


def test_onnx_int8_scorer_matches_torch(tmp_path):
    s = build_split(synthetic_signals())
    std = Standardizer().fit(s.x_train)
    x = std(s.x_test)
    ae = train_ae(std(s.x_train), epochs=20)
    p32 = edge.export_onnx(AEScorer(ae), torch.from_numpy(x[:1]), tmp_path / "ae.onnx")
    p8 = edge.quantize_int8_dynamic(p32, tmp_path / "ae8.onnx")
    ref = ae_scores(ae, x)
    np.testing.assert_allclose(edge.run(edge.session(p32), x), ref, rtol=1e-4, atol=1e-5)
    assert np.corrcoef(edge.run(edge.session(p8), x), ref)[0, 1] > 0.98


# ---------- vision ----------
@pytest.fixture(scope="module")
def feat_model():
    torch.manual_seed(0)
    return vision.PatchFeatures(backbone=resnet18(weights=None)).eval()


def test_patch_features_grid(feat_model):
    f = vision.extract(feat_model, [np.zeros((3, 256, 256), np.float32)])
    assert f.shape == (1, 384, 32, 32)
    assert vision.to_patches(f).shape == (1, 1024, 384)


def test_coreset_covers_clusters():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, 16)) * 20
    x = np.concatenate([c + rng.standard_normal((200, 16)) for c in centers]).astype(np.float32)
    idx = vision.greedy_coreset(x, 5, proj_dim=16)
    assert len(set(np.argmin(((x[idx][:, None] - centers) ** 2).sum(-1), axis=1))) == 5


def test_patchcore_flags_synthetic_defect(feat_model):
    rng = np.random.default_rng(0)
    good = [(rng.standard_normal((3, 256, 256)) * 0.05).astype(np.float32) for _ in range(6)]
    bad = good[0].copy(); bad[:, 100:150, 100:150] += 3.0
    bank = vision.build_memory_bank(vision.to_patches(vision.extract(feat_model, good[:5])), bank_size=500)
    ps = vision.patch_scores(vision.to_patches(vision.extract(feat_model, [good[5], bad])), bank)
    assert ps[1].max() > ps[0].max()
    m = vision.anomaly_maps(ps)
    assert m.shape == (2, 256, 256)
    hot = np.unravel_index(np.argmax(m[1]), m[1].shape)
    assert 80 <= hot[0] <= 170 and 80 <= hot[1] <= 170


def test_visa_split_reader(tmp_path):
    (tmp_path / "split_csv").mkdir()
    (tmp_path / "split_csv" / "1cls.csv").write_text(
        "object,split,label,image,mask\n"
        "pcb1,train,normal,pcb1/Data/Images/Normal/0.JPG,\n"
        "pcb1,test,anomaly,pcb1/Data/Images/Anomaly/1.JPG,pcb1/Data/Masks/Anomaly/1.png\n"
        "cashew,train,normal,cashew/Data/Images/Normal/0.JPG,\n")
    tr, te = vision.visa_split(tmp_path, "pcb1")
    assert len(tr) == 1 and len(te) == 1 and te[0].label == 1 and tr[0].mask is None
