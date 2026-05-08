# Deepfake Detection System Architecture (Group 31)

## 1) System Architecture Overview
The system performs face-focused binary deepfake classification and returns both class and confidence-driven risk level. It is designed for Google Colab execution with data/checkpoints persisted in Google Drive.

```text
[Google Drive Dataset]
        |
        v
[Image Loader: real/, fake/]
        |
        v
[MTCNN Face Detection + Crop]
        |
        v
[Resize 128x128 + Normalize]
        |
        v
[EfficientNet-B0 Backbone]
        |
        v
[Custom Head: 1280->256->1 + Sigmoid]
        |
        +---------------------> [Training: BCELoss + Adam + StepLR + AMP]
        |
        v
[Probability Score (0-1)]
        |
        v
[Decision: REAL/FAKE + LOW/MEDIUM/HIGH Risk]
```

## 2) Module-Level Design
| Module | Input | Process | Output | Technology |
|---|---|---|---|---|
| Data Ingestion | `/data/real`, `/data/fake` images | Read image files and assign labels (`real=0`, `fake=1`) | Tensor-ready samples | Python, PIL, PyTorch Dataset |
| Face Preprocessing | Raw RGB images | Face detection and aligned crop | Face-focused image tensors | facenet-pytorch (MTCNN) |
| Augmentation/Normalization | Cropped faces | Resize to 128x128, tensor conversion, normalize to mean/std 0.5 | Model-ready tensors | torchvision.transforms |
| Split & Loader | Tensor dataset | 80/10/10 split with seed=42 and batching | Train/val/test DataLoaders | torch.utils.data |
| Detector Model | Batch tensor `[B,3,128,128]` | EfficientNet-B0 feature extraction + custom classifier | Fake probability `[B,1]` | torchvision.models, PyTorch |
| Training Engine | Train/val loaders | Forward, BCELoss, AMP backprop, optimization, scheduling | Best checkpoint, metrics history | torch.cuda.amp, Adam, StepLR, wandb |
| Evaluation Engine | Test loader + checkpoint | Compute Accuracy/F1/AUC/Precision/Recall + plots | Metrics and PNG artifacts | scikit-learn, matplotlib, seaborn |
| Confidence Scoring | Sigmoid probability | Threshold to class and risk bands | `label`, `confidence`, `risk_level` | Python inference utility |

## 3) Data Flow with Tensor Shapes
1. **Raw input image**: `H x W x 3` (uint8)
2. **After face crop (MTCNN)**: approx `160 x 160 x 3` (depends on face box)
3. **After resize transform**: `128 x 128 x 3`
4. **After ToTensor**: `[3, 128, 128]` (float32 in [0,1])
5. **After normalize**: `[3, 128, 128]` (centered around 0)
6. **Batch from DataLoader**: `[32, 3, 128, 128]`
7. **Backbone pooled feature**: `[32, 1280]`
8. **Hidden layer**: `[32, 256]`
9. **Output probability**: `[32, 1]`

## 4) Model Layer Specification
| Layer Order | Layer Type | Configuration | Output Shape (for B=32) |
|---|---|---|---|
| 1 | Input | RGB image tensor | `[32, 3, 128, 128]` |
| 2 | EfficientNet-B0 Features | Pretrained ImageNet backbone | `[32, 1280, 4, 4]` (approx) |
| 3 | Global Pool (inside model) | Adaptive pooling | `[32, 1280]` |
| 4 | Dropout | `p=0.4` | `[32, 1280]` |
| 5 | Linear | `1280 -> 256` | `[32, 256]` |
| 6 | ReLU | inplace activation | `[32, 256]` |
| 7 | Linear | `256 -> 1` | `[32, 1]` |
| 8 | Sigmoid | probability mapping | `[32, 1]` |

## 5) Training Configuration
| Parameter | Value |
|---|---|
| Random Seed | 42 |
| Input Size | 128 x 128 |
| Labels | real=0, fake=1 |
| Dataset Target | 10,000 images |
| Split Ratio | 80% train / 10% val / 10% test |
| Batch Size | 32 |
| Epochs | 15 |
| Loss Function | BCELoss |
| Optimizer | Adam (`lr=1e-4`) |
| LR Scheduler | StepLR (`step_size=3`, `gamma=0.5`) |
| Mixed Precision | `torch.cuda.amp` + `GradScaler` |
| Checkpoint | `/content/drive/MyDrive/Deepfake_Group31/checkpoints/best_model.pth` |
| Training Curves | `/content/drive/MyDrive/Deepfake_Group31/outputs/training_curves.png` |
