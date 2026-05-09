# EfficientNet-based Deepfake Detection with Confidence Scoring

**Group Number:** 31  
**Guide:** Prof. Abhishek Majumdar  
**College:** Academy of Technology (MAKAUT Affiliated)

## Team Members
- Rounak Ghosal — Roll No: 169001230071
- Dyutimay Ghosh — Roll No: 16900123177
- Bidisha Maji — Roll No: 16900123170
- Archisman Sinha — Roll No: 16900123021
- Bimal Kr. Mahato — Roll No: 16900123032

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![TorchVision](https://img.shields.io/badge/TorchVision-Models-orange)
![facenet--pytorch](https://img.shields.io/badge/facenet--pytorch-MTCNN-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5c3ee8)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Metrics-f7931e)
![Weights%20%26%20Biases](https://img.shields.io/badge/W%26B-Experiment%20Tracking-yellow)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-f9ab00)

## Repository Structure
- `notebooks/01_preprocessing.ipynb` — Data loading, deterministic split, MTCNN face crop, and preprocessing visualization. **Owner:** Member 1
- `notebooks/02_model_training.ipynb` — EfficientNet-B0 based detector, mixed precision training, checkpointing, wandb logging. **Owner:** Member 2
- `notebooks/03_evaluation.ipynb` — Test metrics, ROC/confusion matrix plots, and confidence-risk inference. **Owner:** Member 3
- `documents/architecture.md` — End-to-end architecture, module design, tensor data flow, and training configuration. **Owner:** Member 4
- `documents/methodology.md` — Model and method justification choices. **Owner:** Member 4
- `requirements.txt` — Python dependencies with pinned versions.

## Google Colab + Google Drive Setup
1. Create a folder in Google Drive: `/content/drive/MyDrive/Deepfake_Group31`.
2. Inside it, prepare dataset folders:
   - `/content/drive/MyDrive/Deepfake_Group31/data/real`
   - `/content/drive/MyDrive/Deepfake_Group31/data/fake`
3. Open each notebook in Google Colab and run cells top-to-bottom.
4. Ensure the first cell mounts Drive and sets:
   - `BASE_DIR = '/content/drive/MyDrive/Deepfake_Group31'`
5. Install dependencies in Colab:
   ```bash
   pip install -r requirements.txt
   ```
6. Run in sequence:
   1. `01_preprocessing.ipynb`
   2. `02_model_training.ipynb`
   3. `03_evaluation.ipynb`

## Notes
- Reproducibility seed is fixed to `42` across notebooks.
- Label mapping is fixed as `real=0`, `fake=1`.
- Best model checkpoint path:
  `/content/drive/MyDrive/Deepfake_Group31/checkpoints/best_model.pth`
