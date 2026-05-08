# Methodology and Design Choices

## 1) Why EfficientNet-B0 over ResNet-50, VGG-16, and Plain CNN
- **Parameter efficiency**: EfficientNet-B0 delivers strong accuracy with fewer parameters than ResNet-50 and far fewer than VGG-16, making it better suited for limited Colab T4 sessions.
- **Balanced scaling**: Compound scaling of depth/width/resolution provides better accuracy-per-FLOP than plain CNN designs that usually require manual architecture tuning.
- **Faster experimentation**: Smaller footprint improves iteration speed, enabling frequent validation/checkpointing during student project timelines.

## 2) Why Transfer Learning instead of Training from Scratch
- **Data efficiency**: Transfer learning reuses robust low/mid-level visual features from ImageNet, reducing dependence on very large domain datasets.
- **Convergence speed**: Fine-tuning converges faster and more stably in Colab than random initialization.
- **Generalization**: Pretrained representations often improve robustness to lighting/compression variations common in manipulated media.

## 3) Why GAN Discriminator-Oriented Detection over Standalone CNN
- **Forgery-sensitive features**: A discriminator-style detector focuses on subtle synthesis artifacts and inconsistencies that generic classifiers may underemphasize.
- **Adversarial framing**: GAN-origin manipulations are naturally aligned with discriminator-driven detection objectives.
- **Practical extensibility**: The detector can be improved later with adversarially generated hard negatives without redesigning the full pipeline.

## 4) Why Sigmoid Confidence Score instead of Softmax
- **Binary task fit**: Deepfake detection here is binary (`real` vs `fake`), where a single sigmoid output is sufficient and interpretable.
- **Direct risk mapping**: A single probability in `[0,1]` maps cleanly to LOW/MEDIUM/HIGH risk thresholds.
- **Simpler calibration**: Confidence interpretation and threshold tuning are straightforward compared with two-logit softmax outputs.

## 5) Why MTCNN over Haar Cascade and Dlib
- **Higher robustness**: MTCNN handles pose/illumination variation better than Haar Cascade in unconstrained social-media-like imagery.
- **Deep learning based localization**: It generally provides better face crops for downstream neural models than classic handcrafted methods.
- **Colab compatibility**: `facenet-pytorch` offers an integrated, GPU-friendly implementation suitable for T4 workflows.
