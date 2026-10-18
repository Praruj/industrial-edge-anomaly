"""Vision track: PatchCore on VisA categories, then the same detector as int8 ONNX.

Usage: python scripts/run_vision.py [--root data/raw/visa/VisA] [--categories pcb1 capsules cashew]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iea import edge  # noqa: E402
from iea.vision import (PatchFeatures, anomaly_maps, coreset_order, extract, load_image,  # noqa: E402
                        load_mask, patch_scores, to_patches, visa_split)

BANK_SIZES = (1_000, 2_500, 5_000, 10_000)


def pixel_auroc(maps: np.ndarray, masks: np.ndarray, max_pixels: int = 2_000_000, seed: int = 0) -> float:
    y, s = masks.ravel(), maps.ravel()
    if len(y) > max_pixels:
        i = np.random.default_rng(seed).choice(len(y), max_pixels, replace=False)
        y, s = y[i], s[i]
    return float(roc_auc_score(y, s))


def score(patches, bank, labels, masks):
    ps = patch_scores(patches, bank)
    img = ps.max(axis=1)
    return {"image_auroc": float(roc_auc_score(labels, img)),
            "pixel_auroc": pixel_auroc(anomaly_maps(ps), masks)}, ps, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/visa/VisA")
    ap.add_argument("--categories", nargs="+", default=["pcb1", "capsules", "cashew"])
    ap.add_argument("--out", default="outputs/vision")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    model = PatchFeatures().eval()
    fp32 = edge.export_onnx(model, torch.zeros(1, 3, 256, 256), out / "patchcore_resnet18_fp32.onnx")
    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    results.update({"backbone": "resnet18 (ImageNet), layer2+layer3", "image_size": 256,
                    "bank_sizes": list(BANK_SIZES), "edge_device": edge.device_info()})
    results.setdefault("categories", {})

    for cat in a.categories:
        train, test = visa_split(root, cat)
        print(f"{cat}: {len(train)} train (good only), {len(test)} test", flush=True)
        tr_imgs = [load_image(s.image) for s in train]
        te_imgs = [load_image(s.image) for s in test]
        labels = np.array([s.label for s in test])
        masks = np.stack([load_mask(s.mask) for s in test])
        r = {"n_train": len(train), "n_test": len(test), "n_defect": int(labels.sum())}

        # fp32 reference with the largest bank
        pool, order = coreset_order(to_patches(extract(model, tr_imgs)), bank_size=max(BANK_SIZES))
        bank32 = pool[order]
        r_fp32, *_ = score(to_patches(extract(model, te_imgs)), bank32, labels, masks)
        sess32 = edge.session(fp32, threads=1)
        one = te_imgs[0][None]
        r["fp32"] = {**r_fp32, "model_mb": edge.size_mb(fp32), "bank_size": len(bank32),
                     **edge.latency_ms(lambda: patch_scores(to_patches(edge.run(sess32, one)), bank32), iters=30)}

        # int8: calibrate on good training images only, rebuild the bank from int8 features
        int8 = edge.quantize_int8_static(fp32, out / f"patchcore_resnet18_int8_{cat}.onnx",
                                         [tr_imgs[i][None] for i in range(0, len(tr_imgs), max(1, len(tr_imgs) // 32))])
        sess = edge.session(int8, threads=1)
        q = lambda ims: to_patches(np.concatenate([edge.run(sess, im[None]) for im in ims]))  # noqa: E731
        pool8, order8 = coreset_order(q(tr_imgs), bank_size=max(BANK_SIZES))
        te8 = q(te_imgs)
        r["int8_backbone_only"] = {"model_mb": edge.size_mb(int8),
                                   **edge.latency_ms(lambda: edge.run(sess, one), iters=30)}
        r["int8_by_bank_size"] = {}
        for k in BANK_SIZES:
            bank = pool8[order8[:k]]
            rk, ps, img = score(te8, bank, labels, masks)
            rk.update(edge.latency_ms(lambda: patch_scores(to_patches(edge.run(sess, one)), bank), iters=30))
            rk["bank_mb_fp16"] = bank.size * 2 / 1e6
            r["int8_by_bank_size"][str(k)] = rk
            print(cat, "int8 bank", k, rk, flush=True)
            if k == max(BANK_SIZES):
                np.save(out / f"{cat}_bank_int8.npy", bank.astype(np.float16))
                o = np.argsort(-img)
                keep = [int(o[0]), int(o[len(o) // 2]), int(np.where(labels == 0)[0][0])]
                np.savez_compressed(out / f"{cat}_examples.npz", idx=np.array(keep), scores=ps[keep],
                                    labels=labels[keep], paths=np.array([str(test[i].image) for i in keep]),
                                    masks=masks[keep])

        results["categories"][cat] = r
        res_path.write_text(json.dumps(results, indent=2))  # save after each category
        print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    main()
