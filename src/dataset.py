import cv2
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG


def get_transforms(img_size=CFG.IMG_SIZE, mode="train"):
    norm = A.Normalize(mean=CFG.IMAGENET_MEAN, std=CFG.IMAGENET_STD)
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
            norm,
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        norm,
        ToTensorV2(),
    ])


class DeepfakeImageDataset(Dataset):
    """
    CSV-based dataset. CSV columns: path, label, video_id, source.
    Label convention: 0 = Real, 1 = Fake.
    Returns (image_tensor, label_float, video_id).
    """
    def __init__(self, csv_path, frames_dir=CFG.FRAMES_DIR, transform=None):
        self.df = pd.read_csv(csv_path, dtype={"video_id": str})
        self.frames_dir = Path(frames_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.frames_dir / row["path"]

        # CSV stores only the filename; images are inside train/val/test folders
        if not img_path.exists():
            split = self.csv_path.stem.replace("_faces", "")
            img_path = self.frames_dir / split / row["path"]

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read {row['path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return img, label, row["video_id"]


def compute_pos_weight(csv_path):
    """pos_weight for BCEWithLogitsLoss = n_real / n_fake (positive class = fake)."""
    df = pd.read_csv(csv_path)
    n_fake = int((df["label"] == 1).sum())
    n_real = int((df["label"] == 0).sum())
    return n_real / max(n_fake, 1)