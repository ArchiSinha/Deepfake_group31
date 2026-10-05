import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import CFG
from dataset import DeepfakeImageDataset, get_transforms, compute_pos_weight
from discriminator import Discriminator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(labels, scores):
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Returns (frame_auc, video_auc). Scores are P(fake), labels 0=real 1=fake."""
    model.eval()
    labels, scores, vids = [], [], []
    for i, (x, y, vid) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x).squeeze(1)
        scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
        labels.extend(y.int().tolist())
        vids.extend(list(vid))

    frame_auc = safe_auc(labels, scores)

    per_video = defaultdict(list)
    video_label = {}
    for v, s, l in zip(vids, scores, labels):
        per_video[v].append(s)
        video_label[v] = l
    v_ids = list(per_video.keys())
    video_auc = safe_auc(
        [video_label[v] for v in v_ids],
        [float(np.mean(per_video[v])) for v in v_ids],
    )
    return frame_auc, video_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=CFG.NUM_WORKERS)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Smoke test: limit batches per epoch")
    args = parser.parse_args()

    set_seed(CFG.SEED)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - training on CPU would take hours. Stop and fix torch.")
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    smoke = args.max_batches is not None
    prefix = "smoke_" if smoke else ""
    os.makedirs(CFG.CKPT_DIR, exist_ok=True)

    train_ds = DeepfakeImageDataset(CFG.TRAIN_CSV, transform=get_transforms(mode="train"))
    val_ds = DeepfakeImageDataset(CFG.VAL_CSV, transform=get_transforms(mode="val"))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=smoke,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = Discriminator(backbone=CFG.BACKBONE, pretrained=True).to(device)

    pos_weight = compute_pos_weight(CFG.TRAIN_CSV)
    print(f"pos_weight (n_real/n_fake): {pos_weight:.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_auc, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        for i, (x, y, _) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} batch {i + 1}/{len(train_loader)} loss {running / n_seen:.4f}")
        scheduler.step()

        frame_auc, video_auc = evaluate(model, val_loader, device, args.max_batches)
        train_loss = running / max(n_seen, 1)
        mins = (time.time() - t0) / 60
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | "
              f"val AUC frame {frame_auc:.4f} video {video_auc:.4f} | {mins:.1f} min")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_auc_frame": frame_auc, "val_auc_video": video_auc})
        with open(os.path.join(CFG.CKPT_DIR, prefix + "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_auc": frame_auc,
                "backbone": CFG.BACKBONE, "img_size": CFG.IMG_SIZE}
        torch.save(ckpt, os.path.join(CFG.CKPT_DIR, prefix + "last.pt"))

        score = frame_auc if frame_auc == frame_auc else -1.0  # treat NaN as -1
        if score > best_auc + 1e-4:
            best_auc, bad_epochs = score, 0
            torch.save(ckpt, os.path.join(CFG.CKPT_DIR, prefix + "best.pt"))
            print(f"  new best val AUC {best_auc:.4f} - saved best checkpoint")
        else:
            bad_epochs += 1
            if bad_epochs >= CFG.PATIENCE:
                print(f"Early stopping: no val AUC improvement for {CFG.PATIENCE} epochs")
                break

    print("Done. Best val frame AUC:", best_auc)


if __name__ == "__main__":
    main()