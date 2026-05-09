import torch

class CFG:
    # Local paths
    DATA_ROOT    = "C:/deepfake_project/data"
    REAL_DIR     = f"{DATA_ROOT}/real"
    FAKE_DIR     = f"{DATA_ROOT}/fake"
    CKPT_DIR     = "C:/deepfake_project/checkpoints"

    # Training
    IMG_SIZE          = 256
    BATCH_SIZE        = 8      # increased since dataset is smaller now
    NUM_EPOCHS        = 50
    LR_G              = 2e-4
    LR_D              = 2e-4
    BETA1             = 0.5
    BETA2             = 0.999
    LAMBDA_L1         = 10.0
    SAVE_EVERY        = 5
    MAX_SAMPLES       = 10000  # cap per class

    # Video
    FRAMES_PER_CLIP   = 8

    # Device
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
    SEED              = 42