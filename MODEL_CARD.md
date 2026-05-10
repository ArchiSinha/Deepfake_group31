# Deepfake Detection Model

## Model Name
`deepfake_detector_celebdfv2_efficientnetb4_96acc`

## Architecture
- **Backbone:** EfficientNet-B4
- **Head:** Linear(1792 → 512 → 1)
- **Framework:** PyTorch

## Dataset
- **Name:** Celeb-DF-v2
- **Real frames:** 3993
- **Fake frames:** 4000
- **Total:** 7993 frames

## Performance
| Metric | Score |
|--------|-------|
| Accuracy | 96.55% |
| AUC-ROC | 96.55% |
| F1 Score | 96.52% |
| Precision | 96.52% |
| Recall | 96.52% |

## Training Details
| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch Size | 16 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss | BCEWithLogitsLoss |
| Image Size | 256x256 |

## Saved Formats
- `.pth` — Full model with metadata
- `_weights_only.pth` — Weights only
- `.onnx` — Universal format

## Date Trained
2026-05-10
