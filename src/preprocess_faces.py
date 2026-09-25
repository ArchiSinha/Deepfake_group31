"""
Face Detection and Cropping for Deepfake Detection Pipeline

Detects faces using MTCNN, crops with margin, and resizes to 224×224.
Path-agnostic: can run locally or on Colab with Drive-mounted data.

Usage:
    python src/preprocess_faces.py \
        --frames_dir data/frames \
        --out_dir data/frames_cropped \
        --manifest data/face_crop_manifest.json \
        --audit_csv data/face_crop_audit.csv

Author: Group 31
Date: 2026-09-25
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


# ============================================================================
# Pure Utility Functions (GPU-free, testable)
# ============================================================================


def video_id_from_name(filename: str) -> str:
    """
    Extract video ID from frame filename.

    Args:
        filename: Frame filename (e.g., "id5_id1_0003_f2.jpg")

    Returns:
        Video ID (e.g., "id5_id1_0003")

    Examples:
        >>> video_id_from_name("id5_id1_0003_f2.jpg")
        'id5_id1_0003'
        >>> video_id_from_name("123_f0.jpg")
        '123'
    """
    stem = Path(filename).stem
    video_id, _, _ = stem.rpartition("_f")
    return video_id if video_id else stem


def expand_box(
    box: Tuple[float, float, float, float],
    margin: float,
    frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Expand bounding box by margin factor around center and make square.

    Does NOT clamp to frame bounds — returns the ideal square box even if it
    extends beyond the frame. clamp_and_pad() handles the actual clamping and padding.

    Args:
        box: (x1, y1, x2, y2) in pixels
        margin: Expansion factor (e.g., 1.3 = 30% expansion)
        frame_shape: (height, width) of source frame (unused, kept for API compatibility)

    Returns:
        (x1, y1, x2, y2) expanded and squared, possibly with negative coords or
        coords exceeding frame bounds

    Examples:
        >>> expand_box((10, 10, 60, 60), 1.3, (100, 100))
        (2, 2, 68, 68)
    """
    x1, y1, x2, y2 = box
    h, w = frame_shape  # Kept for API compatibility, not used

    # Current box dimensions
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    box_w = x2 - x1
    box_h = y2 - y1

    # Expand by margin
    new_w = box_w * margin
    new_h = box_h * margin

    # Make square using longer side
    side = max(new_w, new_h)

    # Compute new box centered on (cx, cy)
    x1_new = cx - side / 2
    y1_new = cy - side / 2
    x2_new = cx + side / 2
    y2_new = cy + side / 2

    # Return without clamping — clamp_and_pad handles bounds
    return int(round(x1_new)), int(round(y1_new)), int(round(x2_new)), int(round(y2_new))


def clamp_and_pad(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    target_size: int
) -> np.ndarray:
    """
    Crop frame to box with edge padding if box exceeds frame bounds, resize to square.

    Args:
        frame: Source frame (H, W, 3) in BGR
        box: (x1, y1, x2, y2) possibly exceeding frame bounds
        target_size: Output size (e.g., 224)

    Returns:
        Cropped and resized frame (target_size, target_size, 3) in BGR
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    # Compute overlap with frame
    x1_src = max(0, x1)
    y1_src = max(0, y1)
    x2_src = min(w, x2)
    y2_src = min(h, y2)

    # Compute padding needed
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    # Crop valid region
    crop = frame[y1_src:y2_src, x1_src:x2_src]

    # Apply padding (replicate edge pixels)
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REPLICATE
        )

    # Resize to target (use INTER_AREA for downscaling, INTER_CUBIC for upscaling)
    side = crop.shape[0]  # Should be square after padding
    interp = cv2.INTER_AREA if side > target_size else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (target_size, target_size), interpolation=interp)

    return resized


def atomic_save(image: np.ndarray, dest_path: Path, quality: int = 95) -> None:
    """
    Save image atomically (temp file then rename) to prevent partial writes.

    Args:
        image: Image array in BGR (OpenCV format)
        dest_path: Final output path
        quality: JPEG quality (1-100)

    Raises:
        IOError: If save fails
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory with proper image extension
    # so cv2.imwrite can determine the encoder
    suffix = dest_path.suffix if dest_path.suffix else '.jpg'
    with tempfile.NamedTemporaryFile(
        mode='wb',
        delete=False,
        dir=dest_path.parent,
        suffix=suffix
    ) as tmp:
        tmp_path = Path(tmp.name)
        success = cv2.imwrite(
            str(tmp_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        if not success:
            tmp_path.unlink(missing_ok=True)
            raise IOError(f"cv2.imwrite failed for {dest_path}")

    # Atomic rename
    tmp_path.replace(dest_path)


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_library_versions() -> Dict[str, str]:
    """Get versions of key libraries."""
    import facenet_pytorch
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "facenet_pytorch": facenet_pytorch.__version__,
        "cv2": cv2.__version__,
        "numpy": np.__version__,
    }


# ============================================================================
# Face Detection and Cropping
# ============================================================================


def process_frame(
    frame_path: Path,
    out_path: Path,
    detector: MTCNN,
    min_prob: float,
    margin: float,
    target_size: int,
    resume: bool,
    dry_run: bool
) -> Tuple[str, Dict]:
    """
    Process a single frame: detect face, crop, resize, save.

    This is a convenience wrapper around process_frame_batch for single-frame processing.
    Used primarily for testing and compatibility.

    Args:
        frame_path: Input frame path
        out_path: Output crop path
        detector: MTCNN detector instance
        min_prob: Minimum detection probability threshold
        margin: Box expansion factor
        target_size: Output image size (pixels)
        resume: Skip if output already exists
        dry_run: Don't write files

    Returns:
        (status, metadata) where status is one of:
            "success", "no_face", "low_confidence", "read_error",
            "detector_error", "save_error", "skipped"
        metadata contains: det_prob, box, crop_side_px, n_faces
    """
    # Call batched version with single frame
    batch_results = process_frame_batch(
        frame_batch=[(frame_path, out_path)],
        detector=detector,
        min_prob=min_prob,
        margin=margin,
        target_size=target_size,
        resume=resume,
        dry_run=dry_run
    )
    return batch_results[0]


def process_frame_batch(
    frame_batch: List[Tuple[Path, Path]],
    detector: MTCNN,
    min_prob: float,
    margin: float,
    target_size: int,
    resume: bool,
    dry_run: bool
) -> List[Tuple[str, Dict]]:
    """
    Process a batch of frames with batched MTCNN inference.

    Args:
        frame_batch: List of (frame_path, out_path) tuples
        detector: MTCNN detector instance
        min_prob: Minimum detection probability threshold
        margin: Box expansion factor
        target_size: Output image size (pixels)
        resume: Skip if output already exists
        dry_run: Don't write files

    Returns:
        List of (status, metadata) tuples, one per frame in batch
    """
    results = []

    # Prepare batch data
    batch_data = []
    for frame_path, out_path in frame_batch:
        metadata = {
            "det_prob": None,
            "box": None,
            "crop_side_px": None,
            "n_faces": 0
        }

        # Resume check
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            results.append(("skipped", metadata))
            batch_data.append(None)
            continue

        try:
            # Read frame
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                results.append(("read_error", metadata))
                batch_data.append(None)
                continue

            # Convert BGR to RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            batch_data.append({
                "frame_bgr": frame_bgr,
                "pil_image": pil_image,
                "frame_path": frame_path,
                "out_path": out_path,
                "metadata": metadata
            })
        except Exception as e:
            logging.error(f"Error reading {frame_path.name}: {e}")
            results.append(("read_error", metadata))
            batch_data.append(None)

    # Extract valid images for batched detection
    valid_indices = [i for i, data in enumerate(batch_data) if data is not None]
    valid_images = [batch_data[i]["pil_image"] for i in valid_indices]

    # Run batched MTCNN detection
    # facenet-pytorch MTCNN.detect() accepts a list of PIL Images of different sizes
    batch_boxes = [None] * len(valid_images)
    batch_probs = [None] * len(valid_images)

    if valid_images:
        try:
            # Batched detection: returns lists of boxes and probs, one per image
            result = detector.detect(valid_images)

            # Handle both tuple (boxes, probs) and list of tuples formats
            if isinstance(result, tuple) and len(result) == 2:
                boxes_result, probs_result = result

                # If single image was passed and result is not a list, wrap it
                if not isinstance(boxes_result, list):
                    batch_boxes = [boxes_result]
                    batch_probs = [probs_result]
                else:
                    batch_boxes = boxes_result
                    batch_probs = probs_result
            else:
                # Unexpected format
                batch_boxes = [None] * len(valid_images)
                batch_probs = [None] * len(valid_images)

        except Exception as e:
            logging.warning(f"Batched detector failed: {e}")
            # Fall back to None for all
            batch_boxes = [None] * len(valid_images)
            batch_probs = [None] * len(valid_images)

    # Process detection results for each valid image
    valid_results_idx = 0
    for i, data in enumerate(batch_data):
        if data is None:
            continue  # Already added to results

        metadata = data["metadata"]
        frame_bgr = data["frame_bgr"]
        frame_path = data["frame_path"]
        out_path = data["out_path"]

        boxes = batch_boxes[valid_results_idx]
        probs = batch_probs[valid_results_idx]
        valid_results_idx += 1

        try:
            # Handle no detections
            if boxes is None or len(boxes) == 0:
                results.append(("no_face", metadata))
                continue

            metadata["n_faces"] = len(boxes)

            # Select largest face by area
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            idx = np.argmax(areas)
            box = boxes[idx]
            prob = probs[idx]

            metadata["det_prob"] = float(prob)

            # Check confidence threshold
            if prob < min_prob:
                results.append(("low_confidence", metadata))
                continue

            # Expand and square box
            h, w = frame_bgr.shape[:2]
            expanded_box = expand_box(
                tuple(box.tolist()),
                margin,
                (h, w)
            )
            metadata["box"] = expanded_box
            metadata["crop_side_px"] = expanded_box[2] - expanded_box[0]

            # Crop with padding and resize
            crop = clamp_and_pad(frame_bgr, expanded_box, target_size)

            # Save
            if not dry_run:
                try:
                    atomic_save(crop, out_path, quality=95)
                except IOError as e:
                    logging.warning(f"Save failed for {frame_path.name}: {e}")
                    results.append(("save_error", metadata))
                    continue

            results.append(("success", metadata))

        except Exception as e:
            logging.error(f"Unexpected error processing {frame_path.name}: {e}")
            results.append(("detector_error", metadata))

    return results


def collect_frame_paths(frames_dir: Path) -> List[Path]:
    """Collect all image files from frames directory."""
    extensions = {'.jpg', '.jpeg', '.png'}
    paths = []
    for ext in extensions:
        paths.extend(frames_dir.glob(f"*{ext}"))
        paths.extend(frames_dir.glob(f"*{ext.upper()}"))
    return sorted(paths)


def compute_video_stats(audit_records: List[Dict]) -> Dict[str, Dict]:
    """
    Compute per-video statistics from audit records.

    Returns:
        {video_id: {"total": int, "success": int, "rate": float}}
    """
    video_stats = defaultdict(lambda: {"total": 0, "success": 0})

    for record in audit_records:
        vid = record["video_id"]
        video_stats[vid]["total"] += 1
        if record["status"] == "success":
            video_stats[vid]["success"] += 1

    # Compute success rate
    for vid, stats in video_stats.items():
        stats["rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0

    return dict(video_stats)


def write_manifest(
    manifest_path: Path,
    audit_records: List[Dict],
    failure_counts: Dict[str, int],
    config: Dict,
    video_stats: Dict[str, Dict]
) -> None:
    """Write manifest JSON with summary statistics and run metadata."""
    total = len(audit_records)
    success = sum(1 for r in audit_records if r["status"] == "success")

    # Videos with <50% success rate
    low_success_videos = {
        vid: stats for vid, stats in video_stats.items()
        if stats["rate"] < 0.5 and stats["total"] >= 2
    }

    manifest = {
        "summary": {
            "total_frames": total,
            "success": success,
            "failure": total - success,
            "success_rate": success / total if total > 0 else 0.0
        },
        "failures_by_reason": dict(failure_counts),
        "videos_below_50pct": {
            vid: {
                "success": stats["success"],
                "total": stats["total"],
                "rate": round(stats["rate"], 3)
            }
            for vid, stats in sorted(
                low_success_videos.items(),
                key=lambda x: x[1]["rate"]
            )
        },
        "config": config,
        "git_commit": get_git_commit(),
        "library_versions": get_library_versions()
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def write_audit_csv(audit_path: Path, audit_records: List[Dict]) -> None:
    """Write audit CSV with per-frame details."""
    import csv

    with open(audit_path, 'w', newline='') as f:
        fieldnames = [
            "filename", "video_id", "status", "det_prob",
            "box", "crop_side_px", "n_faces"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_records)


# ============================================================================
# Main Pipeline
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MTCNN face detection and cropping for deepfake detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # I/O paths
    parser.add_argument(
        "--frames_dir", type=str, required=True,
        help="Input directory containing extracted video frames"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for cropped faces"
    )
    parser.add_argument(
        "--manifest", type=str, default="face_crop_manifest.json",
        help="Path to output manifest JSON"
    )
    parser.add_argument(
        "--audit_csv", type=str, default="face_crop_audit.csv",
        help="Path to output audit CSV"
    )

    # Processing parameters
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for MTCNN face detection (processes multiple images together)"
    )
    parser.add_argument(
        "--min_prob", type=float, default=0.90,
        help="Minimum face detection probability threshold"
    )
    parser.add_argument(
        "--margin", type=float, default=1.3,
        help="Bounding box expansion factor"
    )
    parser.add_argument(
        "--size", type=int, default=224,
        help="Output crop size (pixels, square)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of parallel workers (0 = single-threaded)"
    )

    # Control flags
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip frames that already have non-empty output files"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run detection without saving crops (for testing)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Convert paths
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    audit_path = Path(args.audit_csv)

    # Validate input
    if not frames_dir.exists():
        logging.error(f"Frames directory not found: {frames_dir}")
        sys.exit(1)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    detector = MTCNN(
        keep_all=True,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        device=device
    )

    # Collect frames
    frame_paths = collect_frame_paths(frames_dir)
    logging.info(f"Found {len(frame_paths)} frames in {frames_dir}")

    if len(frame_paths) == 0:
        logging.warning("No frames found, exiting")
        sys.exit(0)

    # Process frames in batches
    audit_records = []
    failure_counts = Counter()

    # Create batches of (frame_path, out_path) tuples
    batch_size = args.batch_size
    num_batches = (len(frame_paths) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(frame_paths))
        batch_frame_paths = frame_paths[start_idx:end_idx]

        # Prepare batch
        frame_batch = [(fp, out_dir / fp.name) for fp in batch_frame_paths]

        # Process batch
        batch_results = process_frame_batch(
            frame_batch=frame_batch,
            detector=detector,
            min_prob=args.min_prob,
            margin=args.margin,
            target_size=args.size,
            resume=args.resume,
            dry_run=args.dry_run
        )

        # Record results
        for (frame_path, out_path), (status, metadata) in zip(frame_batch, batch_results):
            if status != "success" and status != "skipped":
                failure_counts[status] += 1

            # Record audit entry
            audit_records.append({
                "filename": frame_path.name,
                "video_id": video_id_from_name(frame_path.name),
                "status": status,
                "det_prob": metadata["det_prob"],
                "box": str(metadata["box"]) if metadata["box"] else None,
                "crop_side_px": metadata["crop_side_px"],
                "n_faces": metadata["n_faces"]
            })

    # Compute video-level statistics
    video_stats = compute_video_stats(audit_records)

    # Write manifest
    config = {
        "frames_dir": str(frames_dir),
        "out_dir": str(out_dir),
        "min_prob": args.min_prob,
        "margin": args.margin,
        "target_size": args.size,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "seed": args.seed
    }

    if not args.dry_run:
        write_manifest(manifest_path, audit_records, failure_counts, config, video_stats)
        write_audit_csv(audit_path, audit_records)
        logging.info(f"Manifest written to {manifest_path}")
        logging.info(f"Audit CSV written to {audit_path}")

    # Print summary
    success = sum(1 for r in audit_records if r["status"] == "success")
    total = len(audit_records)
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing complete: {success}/{total} frames succeeded")
    logging.info(f"Success rate: {100*success/total:.1f}%")
    logging.info(f"Failures by reason: {dict(failure_counts)}")
    logging.info(f"Videos with <50% success: {len(video_stats) - sum(1 for v in video_stats.values() if v['rate'] >= 0.5)}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
