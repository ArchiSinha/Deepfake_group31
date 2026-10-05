import torch


class CFG:
    # ---- Data (run everything from the repo root) ----
    FRAMES_DIR = "data/frames_cropped"
    TRAIN_CSV  = "data/train.csv"
    VAL_CSV    = "data/val.csv"
    TEST_CSV   = "data/test.csv"
    CKPT_DIR   = "checkpoints"

    # ---- Label convention: 0 = Real, 1 = Fake. Model score = P(fake) ----
    IMG_SIZE   = 224
    BACKBONE   = "efficientnet_b0"
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # ---- Training (Stage 1: supervised classifier) ----
    BATCH_SIZE   = 32      # drop to 16 if CUDA out-of-memory
    NUM_EPOCHS   = 10
    LR           = 2e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE     = 3       # early stopping on val AUC
    NUM_WORKERS  = 4       # set to 0 if DataLoader errors on Windows
    POS_WEIGHT   = None    # computed from train.csv in train_classifier.py

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED   = 42


class LegacyGANCFG:
    """Old GAN hyperparameters, kept only for generator.py / old train.py. Not used in Stage 1."""
    LR_G      = 5e-4
    LR_D      = 2e-4
    BETA1     = 0.5
    BETA2     = 0.999
    LAMBDA_L1 = 10.0