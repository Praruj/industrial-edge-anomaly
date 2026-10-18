"""Figures and a results summary from outputs/*/results.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iea.vision import IMG, anomaly_maps  # noqa: E402

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
INK, GOLD, GREY = "#1f1f1f", "#b8893f", "#9a9a9a"


def vision_examples(res):
    cats = list(res["categories"])
    fig, axes = plt.subplots(len(cats), 3, figsize=(9, 3.1 * len(cats)))
    axes = np.atleast_2d(axes)
    for r, cat in enumerate(cats):
        ex = np.load(OUT / "vision" / f"{cat}_examples.npz")
        maps = anomaly_maps(ex["scores"])
        # colour only what stands out: from the good part's 99th percentile up to the top score
        lo = float(np.percentile(maps[2], 99))
        hi = max(float(maps.max()), lo + 1e-6)
        titles = ["highest score", "median score", "a good part"]
        for c in range(3):
            ax = axes[r, c]
            im = Image.open(str(ex["paths"][c])).convert("RGB").resize((IMG, IMG))
            ax.imshow(im)
            z = np.clip((maps[c] - lo) / (hi - lo), 0, 1)
            rgba = plt.get_cmap("inferno")(z)
            rgba[..., 3] = 0.7 * z
            ax.imshow(rgba)
            if ex["masks"][c].any():
                ax.contour(ex["masks"][c], levels=[0.5], colors="cyan", linewidths=0.8)
            lab = "defect" if ex["labels"][c] else "good"
            ax.set_title(f"{cat}: {titles[c]} ({lab})", fontsize=9)
            ax.axis("off")
    fig.suptitle("PatchCore heatmaps, int8 model (cyan = true defect outline; colour = above a good part's 99th percentile)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "vision_examples.png", dpi=130)
    plt.close(fig)


def edge_tradeoff(vis, vib):
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    shades = ["#1f1f1f", "#b8893f", "#6b8fb5"]
    for (cat, r), col in zip(vis["categories"].items(), shades):
        ks = sorted(r["int8_by_bank_size"], key=int)
        xs = [r["int8_by_bank_size"][k]["p50_ms"] for k in ks]
        ys = [r["int8_by_bank_size"][k]["image_auroc"] for k in ks]
        ax.plot(xs, ys, "-s", color=col, ms=5, label=f"{cat} int8")
        for x, y, k in zip(xs, ys, ks):
            ax.annotate(f"{int(k) / 1000:g}k", (x, y), xytext=(3, 4), textcoords="offset points",
                        fontsize=7, color=GREY)
        ax.scatter(r["fp32"]["p50_ms"], r["fp32"]["image_auroc"], marker="o", s=55,
                   facecolors="none", edgecolors=col)
    ax.scatter([], [], marker="o", facecolors="none", edgecolors=GREY, label="fp32, 10k bank")
    ax.axvline(vis["categories"][next(iter(vis["categories"]))]["int8_backbone_only"]["p50_ms"],
               color=GREY, ls=":", lw=1)
    ax.text(0.01, 0.02, "dotted: int8 backbone alone", transform=ax.transAxes, fontsize=7, color=GREY)
    ax.set_xlabel("latency per image, 1 CPU thread (ms, median)")
    ax.set_ylabel("image AUROC")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Memory-bank size sets the speed; labels = bank size", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "vision_edge_tradeoff.png", dpi=130)
    plt.close(fig)


def vibration_bars(vib):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    u = vib["unseen_load_3hp"]
    settings = [("all loads seen", vib["seen_loads"]["autoencoder"]["seed0"]),
                ("3 HP unseen", u["autoencoder"]["seed0"]),
                (f"3 HP unseen, threshold re-set\nfrom {u['recalibrated_threshold']['healthy_clip_seconds']} s of healthy data",
                 u["recalibrated_threshold"]["autoencoder"])]
    conds = ["inner_race", "ball", "outer_race"]
    w = 0.26
    for i, (lab, d) in enumerate(settings):
        vals = [d["detection_rate"][c] for c in conds] + [d["false_alarm_rate"]]
        ax.bar(np.arange(4) + (i - 1) * w, vals, w, label=lab, color=[INK, "#c0504d", GOLD][i])
    ax.set_xticks(range(4), ["inner race", "ball", "outer race", "false alarms\n(healthy)"])
    ax.set_ylabel("share of windows flagged")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("Vibration monitor at the 99th-percentile alarm threshold", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "vibration_detection.png", dpi=130)
    plt.close(fig)


def main():
    vis = json.loads((OUT / "vision" / "results.json").read_text())
    vib = json.loads((OUT / "vibration" / "results.json").read_text())
    vision_examples(vis)
    edge_tradeoff(vis, vib)
    vibration_bars(vib)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
