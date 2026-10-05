import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from config import CFG
from dataset import DeepfakeImageDataset, get_transforms
from discriminator import Discriminator

OUT_DIR = "outputs"


@torch.no_grad()
def predict(model, csv_path, device, batch_size=32):
    ds = DeepfakeImageDataset(csv_path, transform=get_transforms(mode="val"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    scores, labels, vids = [], [], []
    for x, y, v in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x).squeeze(1)
        scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
        labels.extend(y.int().tolist())
        vids.extend(list(v))
    return np.array(scores), np.array(labels), np.array(vids)


def to_video_level(scores, labels, vids):
    per_video = defaultdict(list)
    video_label = {}
    for s, l, v in zip(scores, labels, vids):
        per_video[v].append(s)
        video_label[v] = l
    ids = list(per_video.keys())
    v_scores = np.array([np.mean(per_video[v]) for v in ids])
    v_labels = np.array([video_label[v] for v in ids])
    return v_scores, v_labels


def pick_threshold(labels, scores):
    """Threshold maximizing TPR - FPR (Youden's J), chosen on val only."""
    fpr, tpr, thr = roc_curve(labels, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def summarize(labels, scores, thr):
    preds = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0)
    spec = tn / max(tn + fp, 1)
    return {
        "n": int(len(labels)),
        "auc": float(roc_auc_score(labels, scores)),
        "threshold": float(thr),
        "accuracy": float((tp + tn) / len(labels)),
        "balanced_accuracy": float((r + spec) / 2),
        "precision_fake": float(p),
        "recall_fake": float(r),
        "specificity_real": float(spec),
        "f1_fake": float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = torch.load(os.path.join(CFG.CKPT_DIR, "best.pt"), map_location=device)
    model = Discriminator(backbone=ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best.pt (epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f})")

    # 1) Choose thresholds on VAL only, then freeze them
    v_s, v_l, v_ids = predict(model, CFG.VAL_CSV, device)
    vv_s, vv_l = to_video_level(v_s, v_l, v_ids)
    thr_frame = pick_threshold(v_l, v_s)
    thr_video = pick_threshold(vv_l, vv_s)
    with open(os.path.join(CFG.CKPT_DIR, "threshold.json"), "w") as f:
        json.dump({"frame": thr_frame, "video": thr_video}, f, indent=2)
    print(f"Thresholds chosen on val: frame={thr_frame:.4f}  video={thr_video:.4f}")

    # 2) Evaluate on TEST with the frozen thresholds
    t_s, t_l, t_ids = predict(model, CFG.TEST_CSV, device)
    tv_s, tv_l = to_video_level(t_s, t_l, t_ids)
    frame_res = summarize(t_l, t_s, thr_frame)
    video_res = summarize(tv_l, tv_s, thr_video)

    for name, res in (("FRAME-LEVEL", frame_res), ("VIDEO-LEVEL", video_res)):
        print(f"\n=== TEST {name} (n={res['n']}) ===")
        for k, v in res.items():
            if k != "n":
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(os.path.join(OUT_DIR, "test_results.json"), "w") as f:
        json.dump({"frame": frame_res, "video": video_res}, f, indent=2)

    # 3) Plots
    plt.figure(figsize=(6, 6))
    for name, lab, sc, res in (("Frame-level", t_l, t_s, frame_res),
                               ("Video-level", tv_l, tv_s, video_res)):
        fpr, tpr, _ = roc_curve(lab, sc)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - test set (Celeb-DF-v2)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name, res in zip(axes, ("Frame-level", "Video-level"), (frame_res, video_res)):
        cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Real", "Fake"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Real", "Fake"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name} (thr={res['threshold']:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"\nSaved plots and test_results.json to {OUT_DIR}\\")


if __name__ == "__main__":
    main()