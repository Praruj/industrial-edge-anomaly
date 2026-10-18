"""Vibration track: train on healthy windows, flag faults, export the monitor to the edge.

Usage: python scripts/run_vibration.py [--data data/raw/cwru] [--out outputs/vibration]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iea import edge  # noqa: E402
from iea.models import AEScorer, ae_scores, train_ae  # noqa: E402
from iea.vibration import FILES, Standardizer, add_noise, build_split, load_de_signal  # noqa: E402


def evaluate(scores_val, scores_test, split, q=99.0):
    thr = float(np.percentile(scores_val, q))
    alarm = scores_test > thr
    per = {c: float(alarm[split.cond_test == c].mean()) for c in np.unique(split.cond_test)}
    return {
        "auroc": float(roc_auc_score(split.y_test, scores_test)),
        "threshold": thr,
        "false_alarm_rate": per.pop("normal"),
        "detection_rate": per,
    }


def run_setting(signals, test_loads, seeds=(0, 1, 2, 3, 4)):
    split = build_split(signals, test_loads=test_loads)
    std = Standardizer().fit(split.x_train)
    xtr, xva, xte = std(split.x_train), std(split.x_val), std(split.x_test)

    iso = IsolationForest(n_estimators=300, random_state=0).fit(xtr)
    res = {"n_train": int(len(xtr)), "n_test": int(len(xte)),
           "isolation_forest": evaluate(-iso.score_samples(xva), -iso.score_samples(xte), split)}

    runs = []
    for s in seeds:
        ae = train_ae(xtr, seed=s)
        runs.append((evaluate(ae_scores(ae, xva), ae_scores(ae, xte), split), ae))
    aucs = [r["auroc"] for r, _ in runs]
    res["autoencoder"] = {"auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)),
                          "seed0": runs[0][0]}
    if split.x_val_target is not None:
        # Same trained models; only the alarm threshold is re-set from a short healthy clip
        # recorded at the new operating point (what a technician would do at commissioning).
        xvt = std(split.x_val_target)
        res["recalibrated_threshold"] = {
            "healthy_clip_windows": int(len(xvt)),
            "healthy_clip_seconds": round(len(xvt) * 1024 / 12_000, 1),
            "isolation_forest": evaluate(-iso.score_samples(xvt), -iso.score_samples(xte), split),
            "autoencoder": evaluate(ae_scores(runs[0][1], xvt), ae_scores(runs[0][1], xte), split),
        }
    return res, runs[0][1], std, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/cwru")
    ap.add_argument("--out", default="outputs/vibration")
    ap.add_argument("--normal-fs", type=int, default=48_000,
                    help="native sample rate of the CWRU normal-baseline files. Their structural "
                         "resonances only line up with the 12 kHz fault files when read at 48 kHz, "
                         "so they are resampled to 12 kHz. Pass 12000 to reproduce the common mistake.")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    signals = {}
    for fid, (cond, _) in FILES.items():
        fs = a.normal_fs if cond == "normal" else 12_000
        signals[fid] = load_de_signal(Path(a.data) / f"{fid}.mat", native_fs=fs)

    results = {"normal_fs": a.normal_fs}
    results["seen_loads"], ae, std, split = run_setting(signals, None)
    results["unseen_load_3hp"], *_ = run_setting(signals, {3})

    # noise stress test: same pipeline with white noise added to every recording
    sweep = {}
    ref = float(np.mean([np.mean(signals[f] ** 2) for f, (c, _) in FILES.items() if c == "normal"]))
    for snr in (10, 0, -5, -10):
        noisy = {fid: add_noise(x, snr, ref, seed=fid) for fid, x in signals.items()}
        r, *_ = run_setting(noisy, None, seeds=(0, 1, 2))
        sweep[str(snr)] = {"isolation_forest_auroc": r["isolation_forest"]["auroc"],
                           "autoencoder_auroc_mean": r["autoencoder"]["auroc_mean"],
                           "autoencoder_auroc_std": r["autoencoder"]["auroc_std"],
                           "autoencoder_detection": r["autoencoder"]["seed0"]["detection_rate"],
                           "autoencoder_false_alarm": r["autoencoder"]["seed0"]["false_alarm_rate"]}
        print("snr", snr, sweep[str(snr)], flush=True)
    results["noise_sweep_snr_db"] = sweep

    # edge: export the scorer, int8 it, check accuracy and latency on one window at a time
    xte = std(split.x_test)
    fp32 = edge.export_onnx(AEScorer(ae), torch.from_numpy(xte[:1]), out / "vibration_ae_fp32.onnx")
    int8 = edge.quantize_int8_dynamic(fp32, out / "vibration_ae_int8.onnx")
    e = {}
    for name, p in [("fp32", fp32), ("int8", int8)]:
        sess = edge.session(p)
        scores = edge.run(sess, xte)
        one = xte[:1]
        e[name] = {"size_kb": edge.size_mb(p) * 1000,
                   "auroc": float(roc_auc_score(split.y_test, scores)),
                   **edge.latency_ms(lambda: edge.run(sess, one), iters=500)}
    results["edge"] = {"device": edge.device_info(), **e,
                       "window_seconds": 2048 / 12_000,
                       "note": "Latency covers the model only; feature extraction is timed separately."}
    from iea.vibration import features, windows  # feature cost per window, same CPU
    w = windows(signals[97])[:1]
    results["edge"]["feature_extraction"] = edge.latency_ms(lambda: features(w), iters=500)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
