 import torch

class CFG:
    # Paths
    DATA_ROOT    = "/content/drive/MyDrive/deepfake_project/data"
    REAL_DIR     = f"{DATA_ROOT}/real"
    FAKE_DIR     = f"{DATA_ROOT}/fake"
    CKPT_DIR     = "/content/drive/MyDrive/deepfake_project/checkpoints"

    # Training
    IMG_SIZE          = 224
    BATCH_SIZE        = 16
    NUM_EPOCHS        = 50
    LR_G              = 2e-4
    LR_D              = 2e-4
    BETA1             = 0.5
    BETA2             = 0.999
    LAMBDA_L1         = 10.0
    SAVE_EVERY        = 5

    # Video
    FRAMES_PER_CLIP   = 8

    # Device
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
    SEED              = 42
