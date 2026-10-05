import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from facenet_pytorch import MTCNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(ROOT, "checkpoints", "deepfake_detector.onnx")
THR_PATH = os.path.join(ROOT, "checkpoints", "threshold.json")

IMG_SIZE = 224
MARGIN = 1.3
MIN_FACE_CONF = 0.90
N_VIDEO_FRAMES = 8
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
VIDEO_EXT = {"mp4", "mov", "avi", "mkv"}

_state = {}


def _load():
    if not _state:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _state["mtcnn"] = MTCNN(keep_all=True, device=device)
        _state["session"] = ort.InferenceSession(
            ONNX_PATH, providers=["CPUExecutionProvider"])
        with open(THR_PATH) as f:
            _state["thr"] = json.load(f)
    return _state


def crop_face(img_rgb, mtcnn):
    """Largest face -> 1.3x square crop (edge-padded) -> 224x224 -> JPEG q95 round trip.
    Returns (crop_rgb, [x1, y1, x2, y2]) or None if no confident face."""
    boxes, probs = mtcnn.detect(img_rgb)
    if boxes is None:
        return None
    best, best_area = None, 0.0
    for b, p in zip(boxes, probs):
        if p is None or p < MIN_FACE_CONF:
            continue
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area > best_area:
            best, best_area = b, area
    if best is None:
        return None

    x1, y1, x2, y2 = [float(v) for v in best]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * MARGIN / 2
    L, T = int(round(cx - half)), int(round(cy - half))
    R, B = int(round(cx + half)), int(round(cy + half))
    h, w = img_rgb.shape[:2]
    pad = max(0, -L, -T, R - w, B - h)
    if pad > 0:
        img_rgb = cv2.copyMakeBorder(img_rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        L, T, R, B = L + pad, T + pad, R + pad, B + pad
    crop = img_rgb[T:B, L:R]
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    crop = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return crop, [int(x1), int(y1), int(x2), int(y2)]


def _scores(crops):
    """List of 224x224 RGB uint8 crops -> array of P(fake)."""
    x = (np.stack(crops).astype(np.float32) / 255.0 - MEAN) / STD
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)
    logits = _load()["session"].run(["logit"], {"input": x})[0].reshape(-1)
    return 1.0 / (1.0 + np.exp(-logits))


def _logit(v):
    v = float(np.clip(v, 1e-6, 1 - 1e-6))
    return float(np.log(v / (1 - v)))


def _decide(p_raw, thr):
    """Label by the frozen threshold; fake_score is rescaled so thr -> 0.5."""
    fake_score = 1.0 / (1.0 + np.exp(-(_logit(p_raw) - _logit(thr))))
    is_fake = p_raw >= thr
    return {
        "label": "FAKE" if is_fake else "REAL",
        "fake_score": float(fake_score),
        "confidence": float(fake_score if is_fake else 1 - fake_score),
        "threshold_used": float(thr),
        "p_fake_raw": float(p_raw),
    }


def _sample_video(path, n=N_VIDEO_FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in np.linspace(0, total - 1, n, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def analyze(path):
    """Main entry point. Returns a dict with status 'ok', 'no_face', 'unsupported' or 'error'."""
    t0 = time.time()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in IMAGE_EXT | VIDEO_EXT:
        return {"status": "unsupported",
                "message": "Only images (jpg, png, bmp, webp) and videos (mp4, mov, avi, mkv) can be analysed."}
    try:
        st = _load()
        if ext in IMAGE_EXT:
            media_type, thr = "image", st["thr"]["frame"]
            bgr = cv2.imread(path)
            frames = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)] if bgr is not None else []
        else:
            media_type, thr = "video", st["thr"]["video"]
            frames = _sample_video(path)
        if not frames:
            return {"status": "error", "message": "Could not read the file."}

        crops, boxes = [], []
        for fr in frames:
            out = crop_face(fr, st["mtcnn"])
            if out is not None:
                crops.append(out[0])
                boxes.append(out[1])
        base = {"media_type": media_type, "frames_sampled": len(frames), "faces_used": len(crops)}
        if not crops:
            return {"status": "no_face", "message": "No face was detected, so the file was not classified.",
                    "inference_ms": round((time.time() - t0) * 1000), **base}

        p = _scores(crops)
        result = {"status": "ok", **base, **_decide(float(np.mean(p)), thr),
                  "frame_scores": [round(float(v), 4) for v in p],
                  "face_box": boxes[0] if media_type == "image" else None}
        result["inference_ms"] = round((time.time() - t0) * 1000)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _selftest(n=30):
    """Compare this pipeline (raw frame -> crop -> model) with the stored training crops."""
    import pandas as pd
    st = _load()
    df = pd.read_csv(os.path.join(ROOT, "data", "test.csv")).sample(n, random_state=0)
    diffs, agree, used, skipped = [], 0, 0, 0
    thr = st["thr"]["frame"]
    for _, r in df.iterrows():
        raw = os.path.join(ROOT, "data", "frames", r["path"])
        stored = os.path.join(ROOT, "data", "frames_cropped", r["path"])
        bgr = cv2.imread(raw)
        ref_bgr = cv2.imread(stored)
        if bgr is None or ref_bgr is None:
            skipped += 1
            continue
        out = crop_face(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), st["mtcnn"])
        if out is None:
            skipped += 1
            continue
        p_new, p_ref = _scores([out[0], cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)])
        diffs.append(abs(float(p_new) - float(p_ref)))
        agree += int((p_new >= thr) == (p_ref >= thr))
        used += 1
    print(f"Compared {used} frames (skipped {skipped}).")
    if used:
        print(f"Mean abs diff in P(fake): {np.mean(diffs):.4f}  max: {np.max(diffs):.4f}")
        print(f"Same real/fake decision as stored crops: {agree}/{used}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(analyze(sys.argv[1]), indent=2))
    else:
        _selftest()