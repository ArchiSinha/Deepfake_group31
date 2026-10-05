import os

import numpy as np
import onnxruntime as ort
import torch

from config import CFG
from dataset import DeepfakeImageDataset, get_transforms
from discriminator import Discriminator


def main():
    ckpt = torch.load(os.path.join(CFG.CKPT_DIR, "best.pt"), map_location="cpu")
    model = Discriminator(backbone=ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_path = os.path.join(CFG.CKPT_DIR, "deepfake_detector.onnx")
    dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"Exported to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    # Parity check on 16 real test images: PyTorch vs ONNX Runtime (CPU)
    ds = DeepfakeImageDataset(CFG.TEST_CSV, transform=get_transforms(mode="val"))
    idx = np.linspace(0, len(ds) - 1, 16, dtype=int)
    batch = torch.stack([ds[int(i)][0] for i in idx])
    with torch.no_grad():
        torch_logits = model(batch).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logit"], {"input": batch.numpy()})[0]
    diff = float(np.abs(torch_logits - onnx_logits).max())
    print(f"Max abs logit difference (PyTorch vs ONNX): {diff:.6f}")
    print("PARITY OK" if diff < 1e-3 else "PARITY FAILED - do not use this file")


if __name__ == "__main__":
    main()