import cv2
import numpy as np
import os
from pathlib import Path

VIDEO_DIR  = "D:/datasets/Celeb-DF-v2"
OUTPUT_DIR = "C:/Users/ARCHISHMAN/Deepfake_group31/data/frames"
FRAMES_PER_VIDEO = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
count = 0
video_count = 0

video_paths = sorted(Path(VIDEO_DIR).rglob("*.mp4"))
total_videos = len(video_paths)
print(f"Found {total_videos} videos total")

for vpath in video_paths:
    video_count += 1
    out_paths = [os.path.join(OUTPUT_DIR, f"{vpath.stem}_f{i}.jpg") for i in range(FRAMES_PER_VIDEO)]
    if all(os.path.exists(p) for p in out_paths):
        continue  # already done, resume-safe

    cap = cv2.VideoCapture(str(vpath))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"  WARNING: unreadable video, skipping: {vpath.name}")
        cap.release()
        continue

    indices = np.linspace(0, total - 1, FRAMES_PER_VIDEO, dtype=int)
    for i, fi in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(out_paths[i], frame)
            count += 1
    cap.release()

    if video_count % 100 == 0:
        print(f"  [{video_count}/{total_videos}] videos done, {count} frames so far")

print(f"\nDone! Extracted {count} frames from {video_count} videos -> {OUTPUT_DIR}")