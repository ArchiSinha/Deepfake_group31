"""
Identity-Disjoint Dataset Split Generation for Deepfake Detection Pipeline

Generates leak-free train.csv, val.csv, and test.csv splits for Celeb-DF-v2.
Ensures zero identity leakage across splits:
- Celeb identities (real + synth) are partitioned across train/val/test.
- Celeb-synthesis videos are included ONLY IF both subject identities belong to the same split.
- YouTube-real videos are partitioned at the video level.

Usage:
    python src/make_splits.py \\
        --frames_dir data/frames_cropped \\
        --manifest data/face_crop_manifest.json \\
        --out_dir data \\
        --seed 42

Author: Group 31
Date: 2026-10-03
"""

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Import helpers from preprocess_faces, with fallback for standalone execution
try:
    from src.preprocess_faces import get_git_commit, video_id_from_name
except ImportError:
    try:
        from preprocess_faces import get_git_commit, video_id_from_name
    except ImportError:
        def video_id_from_name(filename: str) -> str:
            stem = Path(filename).stem
            video_id, _, _ = stem.rpartition("_f")
            return video_id if video_id else stem

        def get_git_commit() -> str:
            import subprocess
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass
            return "unknown"


# Default random seed
DEFAULT_SEED = 42

# Regex patterns for video identity parsing
CELEB_SYNTH_PATTERN = re.compile(r"^(id\d+)_(id\d+)_(.+)$")
CELEB_REAL_PATTERN = re.compile(r"^(id\d+)_(.+)$")
YT_REAL_PATTERN = re.compile(r"^\d+$")


# ============================================================================
# Pure Utility Functions (GPU-free, testable in isolation)
# ============================================================================


def parse_filename_metadata(filename: str) -> Optional[Dict]:
    """
    Parse a frame filename to extract video ID, identities, label, and source.

    Identities parsing rules:
    - Celeb-real: "id<N>_<clip>_f<K>.jpg" -> identity = "id<N>", label = 0, source = "celeb_real"
    - YouTube-real: "<number>_f<K>.jpg" -> identity = "yt_<number>", label = 0, source = "yt_real"
    - Celeb-synthesis: "id<A>_id<B>_<clip>_f<K>.jpg" -> identities = ("id<A>", "id<B>"), label = 1, source = "celeb_synth"
      NOTE: UNVERIFIED mapping in Celeb-DF-v2 as to which ID is source vs target. Both identities
      are tracked and must reside in the exact same split to avoid identity leakage.

    Args:
        filename: Frame filename or path (e.g., "id5_id1_0003_f2.jpg")

    Returns:
        Dict with keys: filename, video_id, identities (tuple), label (int), source (str),
        or None if filename does not match any known pattern (unparseable).
    """
    fname = Path(filename).name
    vid_id = video_id_from_name(fname)

    # 1. Celeb-synthesis: id<A>_id<B>_<clip>
    # UNVERIFIED: We have not confirmed from the Celeb-DF-v2 README whether id<A>
    # is the source or target face. For identity-disjoint splitting, both identities
    # are tracked and both must be in the same split for the video to be included.
    m_synth = CELEB_SYNTH_PATTERN.match(vid_id)
    if m_synth:
        id_a, id_b, _ = m_synth.groups()
        return {
            "filename": fname,
            "video_id": vid_id,
            "identities": (id_a, id_b),
            "label": 1,
            "source": "celeb_synth",
        }

    # 2. Celeb-real: id<N>_<clip>
    m_celeb = CELEB_REAL_PATTERN.match(vid_id)
    if m_celeb:
        id_celeb, _ = m_celeb.groups()
        return {
            "filename": fname,
            "video_id": vid_id,
            "identities": (id_celeb,),
            "label": 0,
            "source": "celeb_real",
        }

    # 3. YouTube-real: <number>
    m_yt = YT_REAL_PATTERN.match(vid_id)
    if m_yt:
        yt_id = f"yt_{vid_id}"
        return {
            "filename": fname,
            "video_id": vid_id,
            "identities": (yt_id,),
            "label": 0,
            "source": "yt_real",
        }

    # Unparseable filename
    return None


def identity_pools(filenames: List[str]) -> Dict:
    """
    Build identity pools and video metadata from a list of frame filenames.

    This is a pure function that does not touch disk and can be unit tested directly.

    Args:
        filenames: List of frame filenames or paths

    Returns:
        Dict containing:
            - "celeb_identities": Set of unique Celeb identity strings (e.g. {"id0", "id1"})
            - "yt_videos": Set of unique YouTube video ID strings (e.g. {"001", "123"})
            - "identity_to_videos": Dict mapping identity -> set of video_ids
            - "video_info": Dict mapping video_id -> {
                  "video_id": str,
                  "identities": tuple,
                  "source": str,
                  "label": int,
                  "frames": list of str (filenames)
              }
            - "unparseable_files": List of filenames that could not be parsed
    """
    celeb_identities: Set[str] = set()
    yt_videos: Set[str] = set()
    identity_to_videos: Dict[str, Set[str]] = defaultdict(set)
    video_info: Dict[str, Dict] = {}
    unparseable_files: List[str] = []

    for fn in filenames:
        meta = parse_filename_metadata(fn)
        if meta is None:
            unparseable_files.append(fn)
            continue

        vid_id = meta["video_id"]
        fname = meta["filename"]

        if vid_id not in video_info:
            video_info[vid_id] = {
                "video_id": vid_id,
                "identities": meta["identities"],
                "source": meta["source"],
                "label": meta["label"],
                "frames": [],
            }

        video_info[vid_id]["frames"].append(fname)

        # Track identities and video mappings
        if meta["source"] == "yt_real":
            yt_videos.add(vid_id)
            identity_to_videos[meta["identities"][0]].add(vid_id)
        else:
            for ident in meta["identities"]:
                celeb_identities.add(ident)
                identity_to_videos[ident].add(vid_id)

    return {
        "celeb_identities": celeb_identities,
        "yt_videos": yt_videos,
        "identity_to_videos": dict(identity_to_videos),
        "video_info": video_info,
        "unparseable_files": unparseable_files,
    }


def assign_splits(
    pools: Dict,
    ratios: Union[Tuple[float, float, float], Dict[str, float]],
    seed: int
) -> Dict[str, str]:
    """
    Assign identities and YouTube videos to splits ('train', 'val', 'test').

    This is a pure function testable in isolation.

    Args:
        pools: Dictionary returned by identity_pools()
        ratios: (train_ratio, val_ratio, test_ratio) or {"train": 0.7, "val": 0.15, "test": 0.15}
        seed: Random seed for shuffling

    Returns:
        Dict mapping identity (or yt_ video identifier) -> split name ('train', 'val', 'test')
    """
    if isinstance(ratios, dict):
        train_r = ratios.get("train", 0.70)
        val_r = ratios.get("val", 0.15)
        test_r = ratios.get("test", 0.15)
    else:
        train_r, val_r, test_r = ratios

    rng = random.Random(seed)

    identity_to_split: Dict[str, str] = {}

    # 1. Split Celeb identities
    celeb_ids = sorted(list(pools["celeb_identities"]))
    rng.shuffle(celeb_ids)

    n_celeb = len(celeb_ids)
    n_train_c = int(round(n_celeb * train_r))
    n_val_c = int(round(n_celeb * val_r))
    if n_val_c == 0 and val_r > 0 and n_celeb >= 3:
        n_val_c = 1
    n_test_c = n_celeb - n_train_c - n_val_c
    if n_test_c <= 0 and test_r > 0 and n_celeb >= 3:
        n_test_c = 1
        n_train_c = n_celeb - n_val_c - n_test_c

    for ident in celeb_ids[:n_train_c]:
        identity_to_split[ident] = "train"
    for ident in celeb_ids[n_train_c:n_train_c + n_val_c]:
        identity_to_split[ident] = "val"
    for ident in celeb_ids[n_train_c + n_val_c:]:
        identity_to_split[ident] = "test"

    # 2. Split YouTube-real videos (split at video level, identity is "yt_<vid>")
    yt_vids = sorted(list(pools["yt_videos"]))
    rng.shuffle(yt_vids)

    n_yt = len(yt_vids)
    n_train_yt = int(round(n_yt * train_r))
    n_val_yt = int(round(n_yt * val_r))
    if n_val_yt == 0 and val_r > 0 and n_yt >= 3:
        n_val_yt = 1
    n_test_yt = n_yt - n_train_yt - n_val_yt
    if n_test_yt <= 0 and test_r > 0 and n_yt >= 3:
        n_test_yt = 1
        n_train_yt = n_yt - n_val_yt - n_test_yt

    for vid in yt_vids[:n_train_yt]:
        identity_to_split[f"yt_{vid}"] = "train"
    for vid in yt_vids[n_train_yt:n_train_yt + n_val_yt]:
        identity_to_split[f"yt_{vid}"] = "val"
    for vid in yt_vids[n_train_yt + n_val_yt:]:
        identity_to_split[f"yt_{vid}"] = "test"

    return identity_to_split


def assign_videos_to_splits(
    video_info: Dict[str, Dict],
    identity_to_split: Dict[str, str]
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Assign videos to splits based on identity assignments and exclusion rules.

    - Celeb-real: assigned to the split matching its single identity.
    - YouTube-real: assigned to the split matching its yt_<vid> identity.
    - Celeb-synthesis: assigned to a split ONLY IF BOTH of its identities fall into the same split.
      If its identities span different splits, the video is excluded entirely.

    Args:
        video_info: Dict mapping video_id -> video metadata dict
        identity_to_split: Dict mapping identity -> split name

    Returns:
        (video_splits, cross_split_excluded) where:
            - video_splits is {"train": [...], "val": [...], "test": [...]}
            - cross_split_excluded is a list of excluded Celeb-synthesis video_ids
    """
    video_splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    cross_split_excluded: List[str] = []

    for vid_id, info in sorted(video_info.items()):
        source = info["source"]
        identities = info["identities"]

        if source == "celeb_synth":
            id_a, id_b = identities
            split_a = identity_to_split.get(id_a)
            split_b = identity_to_split.get(id_b)

            # Both identities must belong to the exact same split
            if split_a is not None and split_a == split_b:
                video_splits[split_a].append(vid_id)
            else:
                cross_split_excluded.append(vid_id)

        elif source == "celeb_real":
            id_celeb = identities[0]
            split = identity_to_split.get(id_celeb)
            if split:
                video_splits[split].append(vid_id)

        elif source == "yt_real":
            yt_id = identities[0]
            split = identity_to_split.get(yt_id)
            if split:
                video_splits[split].append(vid_id)

    return video_splits, cross_split_excluded


def split_dataset_with_retry(
    pools: Dict,
    ratios: Tuple[float, float, float],
    initial_seed: int = DEFAULT_SEED,
    min_val_test_videos: int = 30,
    max_attempts: int = 20
) -> Tuple[Dict[str, str], Dict[str, List[str]], List[str], int, int, bool, Dict[str, int]]:
    """
    Perform identity-disjoint splitting with retry loop to satisfy minimum video counts.

    Retries shuffling with seeds (initial_seed, initial_seed+1, ..., initial_seed+max_attempts-1)
    until val and test splits each contain >= min_val_test_videos real AND fake videos.

    Args:
        pools: Identity pools dict from identity_pools()
        ratios: (train_ratio, val_ratio, test_ratio)
        initial_seed: Starting random seed
        min_val_test_videos: Minimum real and fake video count in val and test
        max_attempts: Maximum retry attempts (default 20)

    Returns:
        (identity_to_split, video_splits, cross_split_excluded, final_seed, attempts_used, is_met, counts)
    """
    video_info = pools["video_info"]
    best_attempt_data = None
    best_shortfall = float("inf")

    for attempt in range(max_attempts):
        curr_seed = initial_seed + attempt
        id_to_split = assign_splits(pools, ratios, curr_seed)
        v_splits, excluded = assign_videos_to_splits(video_info, id_to_split)

        val_real = sum(1 for v in v_splits["val"] if video_info[v]["label"] == 0)
        val_fake = sum(1 for v in v_splits["val"] if video_info[v]["label"] == 1)
        test_real = sum(1 for v in v_splits["test"] if video_info[v]["label"] == 0)
        test_fake = sum(1 for v in v_splits["test"] if video_info[v]["label"] == 1)

        counts = {
            "val_real": val_real,
            "val_fake": val_fake,
            "test_real": test_real,
            "test_fake": test_fake,
        }

        shortfall = (
            max(0, min_val_test_videos - val_real)
            + max(0, min_val_test_videos - val_fake)
            + max(0, min_val_test_videos - test_real)
            + max(0, min_val_test_videos - test_fake)
        )

        if shortfall < best_shortfall:
            best_shortfall = shortfall
            best_attempt_data = (id_to_split, v_splits, excluded, curr_seed, attempt + 1, counts)

        if shortfall == 0:
            logging.info(
                f"Split constraints met on attempt {attempt + 1} (seed={curr_seed}): "
                f"val (real={val_real}, fake={val_fake}), test (real={test_real}, fake={test_fake}) "
                f">= min_val_test_videos ({min_val_test_videos})"
            )
            return (id_to_split, v_splits, excluded, curr_seed, attempt + 1, True, counts)

        logging.warning(
            f"Attempt {attempt + 1}/{max_attempts} (seed={curr_seed}) below minimum video threshold: "
            f"val real={val_real}/{min_val_test_videos}, fake={val_fake}/{min_val_test_videos}; "
            f"test real={test_real}/{min_val_test_videos}, fake={test_fake}/{min_val_test_videos}"
        )

    # Fallback to best attempt if threshold was not met after max_attempts
    id_to_split, v_splits, excluded, final_seed, final_attempts, counts = best_attempt_data
    logging.warning(
        f"WARNING: After {max_attempts} attempts, minimum video requirements ({min_val_test_videos} real/fake in val/test) "
        f"were NOT met. Proceeding with best attempt (seed={final_seed}): "
        f"val (real={counts['val_real']}, fake={counts['val_fake']}), test (real={counts['test_real']}, fake={counts['test_fake']})"
    )
    return (id_to_split, v_splits, excluded, final_seed, max_attempts, False, counts)


def load_successful_crops(
    manifest_path: Optional[Union[str, Path]],
    frames_dir: Optional[Path] = None
) -> Optional[Set[str]]:
    """
    Load set of successful crop filenames from manifest JSON or audit CSV.

    Args:
        manifest_path: Path to face_crop_manifest.json or face_crop_audit.csv
        frames_dir: Directory containing cropped frames (fallback/reference)

    Returns:
        Set of valid crop filenames, or None if no filtering should be applied
    """
    if manifest_path is None:
        return None

    path = Path(manifest_path)
    if not path.exists():
        logging.warning(f"Manifest file not found at {path}. Proceeding without manifest filtering.")
        return None

    try:
        if path.suffix.lower() == ".csv":
            successful = set()
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "success":
                        fname = row.get("filename")
                        if fname:
                            successful.add(Path(fname).name)
            logging.info(f"Loaded {len(successful)} successful crops from audit CSV {path}")
            return successful

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            successful = set()
            for item in data:
                if isinstance(item, str):
                    successful.add(Path(item).name)
                elif isinstance(item, dict):
                    if item.get("status") in ("success", None):
                        fname = item.get("filename") or item.get("path") or item.get("file")
                        if fname:
                            successful.add(Path(fname).name)
            logging.info(f"Loaded {len(successful)} successful crops from manifest list {path}")
            return successful

        if isinstance(data, dict):
            for key in ["successful_frames", "success_frames", "successful_crops", "crops", "files"]:
                if key in data and isinstance(data[key], list):
                    successful = {
                        Path(x).name if isinstance(x, str) else Path(x.get("filename", "")).name
                        for x in data[key]
                    }
                    logging.info(f"Loaded {len(successful)} successful crops from manifest key '{key}'")
                    return successful

            if "audit_records" in data and isinstance(data["audit_records"], list):
                successful = {
                    Path(r["filename"]).name
                    for r in data["audit_records"]
                    if isinstance(r, dict) and r.get("status") == "success"
                }
                logging.info(f"Loaded {len(successful)} successful crops from manifest audit records")
                return successful

            # Check if adjacent face_crop_audit.csv exists
            adjacent_audit = path.parent / "face_crop_audit.csv"
            if adjacent_audit.exists():
                logging.info(f"Manifest JSON has summary only; loading successful crops from {adjacent_audit}")
                return load_successful_crops(adjacent_audit)

            logging.info(
                f"Manifest at {path} contains run summary. All physical frames in frames_dir will be considered valid."
            )
            return None

    except Exception as e:
        logging.warning(f"Error reading manifest at {path}: {e}. Proceeding without manifest filtering.")
        return None

    return None


# ============================================================================
# CSV Output & Reporting
# ============================================================================


def write_split_csvs(
    video_splits: Dict[str, List[str]],
    video_info: Dict[str, Dict],
    out_dir: Path,
    frames_dir: Path,
    successful_crops: Optional[Set[str]] = None
) -> Dict[str, List[Dict]]:
    """
    Write train.csv, val.csv, test.csv to out_dir.

    Columns: path, label, video_id, source

    Args:
        video_splits: Dict mapping split name -> list of video_ids
        video_info: Dict mapping video_id -> video metadata dict
        out_dir: Destination directory for CSV files
        frames_dir: Root directory of cropped frames
        successful_crops: Set of valid crop filenames to include (if filtered)

    Returns:
        Dict mapping split name -> list of written row dicts
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    fieldnames = ["path", "label", "video_id", "source"]

    for split in ["train", "val", "test"]:
        csv_path = out_dir / f"{split}.csv"
        rows = []

        for vid_id in sorted(video_splits[split]):
            info = video_info[vid_id]
            for fname in sorted(info["frames"]):
                if successful_crops is not None and fname not in successful_crops:
                    continue

                # path is relative to frames_dir (just the filename if flat directory)
                rows.append({
                    "path": fname,
                    "label": info["label"],
                    "video_id": vid_id,
                    "source": info["source"],
                })

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        split_rows[split] = rows
        logging.info(f"Wrote {len(rows)} frames across {len(video_splits[split])} videos to {csv_path}")

    return split_rows


def compute_split_statistics(
    split_rows: Dict[str, List[Dict]],
    video_splits: Dict[str, List[str]],
    video_info: Dict[str, Dict]
) -> Dict:
    """Compute detailed breakdown statistics per split and per source."""
    stats = {}
    for split in ["train", "val", "test"]:
        rows = split_rows[split]
        vids = video_splits[split]

        source_counts = {
            "celeb_real": {"videos": 0, "frames": 0},
            "yt_real": {"videos": 0, "frames": 0},
            "celeb_synth": {"videos": 0, "frames": 0},
        }

        real_vids = sum(1 for v in vids if video_info[v]["label"] == 0)
        fake_vids = sum(1 for v in vids if video_info[v]["label"] == 1)

        real_frames = sum(1 for r in rows if r["label"] == 0)
        fake_frames = sum(1 for r in rows if r["label"] == 1)

        for v in vids:
            src = video_info[v]["source"]
            if src in source_counts:
                source_counts[src]["videos"] += 1

        for r in rows:
            src = r["source"]
            if src in source_counts:
                source_counts[src]["frames"] += 1

        stats[split] = {
            "total_videos": len(vids),
            "total_frames": len(rows),
            "real_videos": real_vids,
            "real_frames": real_frames,
            "fake_videos": fake_vids,
            "fake_frames": fake_frames,
            "by_source": source_counts,
        }

    return stats


def print_summary_table(
    stats: Dict,
    cross_split_excluded: List[str],
    unparseable_files: List[str],
    initial_seed: int,
    final_seed: int,
    attempts: int,
    min_val_test_videos: int,
    is_met: bool
) -> None:
    """Print formatted ASCII summary table to console."""
    print("\n" + "=" * 78)
    print("                      DATASET SPLIT SUMMARY TABLE")
    print("=" * 78)
    print(f"{'Split':<8} {'Source':<14} {'Class':<8} {'Videos':>10} {'Frames':>12}")
    print("-" * 78)

    for split in ["train", "val", "test"]:
        sp = stats[split]
        by_src = sp["by_source"]

        for src, label_str in [("celeb_real", "Real (0)"), ("yt_real", "Real (0)"), ("celeb_synth", "Fake (1)")]:
            v_cnt = by_src[src]["videos"]
            f_cnt = by_src[src]["frames"]
            print(f"{split:<8} {src:<14} {label_str:<8} {v_cnt:>10d} {f_cnt:>12d}")

        print(f"{'':<8} {'--- SUB-TOTAL':<14} {'Total':<8} {sp['total_videos']:>10d} {sp['total_frames']:>12d}")
        print(f"{'':<8} {'(Real / Fake)':<14} {'':<8} {sp['real_videos']:>4d} / {sp['fake_videos']:<4d} {sp['real_frames']:>5d} / {sp['fake_frames']:<5d}")
        print("-" * 78)

    print("SUMMARY & INTEGRITY CHECK:")
    print(f"  - Celeb-synthesis excluded (cross-split identity): {len(cross_split_excluded)}")
    print(f"  - Unparseable filenames excluded:                 {len(unparseable_files)}")
    print(f"  - Initial seed: {initial_seed} | Final seed used: {final_seed}")
    print(f"  - Shuffling attempts required: {attempts}")
    status_str = "PASSED" if is_met else "WARNING (Threshold not met)"
    print(f"  - Min video threshold (>= {min_val_test_videos} real & fake in val/test): {status_str}")
    print("=" * 78 + "\n")


def write_summary_json(
    summary_path: Path,
    stats: Dict,
    cross_split_excluded: List[str],
    unparseable_files: List[str],
    initial_seed: int,
    final_seed: int,
    attempts: int,
    min_val_test_videos: int,
    is_met: bool,
    ratios: Dict[str, float],
    config: Dict
) -> None:
    """Write comprehensive split_summary.json to out_dir."""
    summary_data = {
        "git_commit": get_git_commit(),
        "seed_initial": initial_seed,
        "seed_final": final_seed,
        "attempts": attempts,
        "min_val_test_videos_required": min_val_test_videos,
        "min_val_test_videos_met": is_met,
        "ratios": ratios,
        "counts": {
            "cross_split_excluded_synth_videos": len(cross_split_excluded),
            "unparseable_filenames": len(unparseable_files),
        },
        "splits": stats,
        "config": config,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logging.info(f"Split summary written to {summary_path}")


# ============================================================================
# Main CLI Pipeline
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate identity-disjoint train/val/test splits for Celeb-DF-v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # I/O paths
    parser.add_argument(
        "--frames_dir", type=str, default="data/frames_cropped",
        help="Path to directory containing face-cropped frames"
    )
    parser.add_argument(
        "--manifest", type=str, default="data/face_crop_manifest.json",
        help="Path to face_crop_manifest.json or face_crop_audit.csv from preprocess_faces.py"
    )
    parser.add_argument(
        "--out_dir", type=str, default="data",
        help="Directory to write train.csv, val.csv, test.csv, and split_summary.json"
    )

    # Split parameters
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for identity shuffling"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.70,
        help="Proportion of identities in training split"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
        help="Proportion of identities in validation split"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15,
        help="Proportion of identities in test split"
    )
    parser.add_argument(
        "--min_val_test_videos", type=int, default=30,
        help="Minimum real AND fake videos required in each of val and test splits"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        logging.error(f"Ratios must sum to 1.0; got {args.train_ratio} + {args.val_ratio} + {args.test_ratio} = {total_ratio}")
        sys.exit(1)

    frames_dir = Path(args.frames_dir)
    manifest_path = Path(args.manifest) if args.manifest else None
    out_dir = Path(args.out_dir)

    if not frames_dir.exists():
        logging.error(f"Frames directory not found: {frames_dir}")
        sys.exit(1)

    # Collect all frame files in frames_dir
    extensions = {".jpg", ".jpeg", ".png"}
    all_frame_paths = [
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    logging.info(f"Found {len(all_frame_paths)} frame image files in {frames_dir}")

    if not all_frame_paths:
        logging.warning("No frame files found in frames_dir, exiting")
        sys.exit(0)

    # Load successful crop list from manifest
    successful_crops = load_successful_crops(manifest_path, frames_dir)

    # Filter frame paths if manifest crop list was loaded
    if successful_crops is not None:
        valid_filenames = [p.name for p in all_frame_paths if p.name in successful_crops]
        logging.info(f"Filtered to {len(valid_filenames)} successful crops from manifest")
    else:
        valid_filenames = [p.name for p in all_frame_paths]

    # Build identity pools
    pools = identity_pools(valid_filenames)
    logging.info(
        f"Parsed {len(pools['video_info'])} videos: {len(pools['celeb_identities'])} Celeb identities, "
        f"{len(pools['yt_videos'])} YouTube videos, {len(pools['unparseable_files'])} unparseable files"
    )

    if pools["unparseable_files"]:
        logging.warning(
            f"Encountered {len(pools['unparseable_files'])} unparseable files (excluded from all splits)."
        )

    # Execute split with retry loop
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    (
        identity_to_split,
        video_splits,
        cross_split_excluded,
        final_seed,
        attempts,
        is_met,
        final_counts
    ) = split_dataset_with_retry(
        pools=pools,
        ratios=ratios,
        initial_seed=args.seed,
        min_val_test_videos=args.min_val_test_videos,
        max_attempts=20
    )

    # Write CSVs
    split_rows = write_split_csvs(
        video_splits=video_splits,
        video_info=pools["video_info"],
        out_dir=out_dir,
        frames_dir=frames_dir,
        successful_crops=successful_crops
    )

    # Compute stats
    stats = compute_split_statistics(
        split_rows=split_rows,
        video_splits=video_splits,
        video_info=pools["video_info"]
    )

    # Write split_summary.json
    config_dict = {
        "frames_dir": str(frames_dir),
        "manifest": str(manifest_path) if manifest_path else None,
        "out_dir": str(out_dir),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "min_val_test_videos": args.min_val_test_videos,
    }
    ratios_dict = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    write_summary_json(
        summary_path=out_dir / "split_summary.json",
        stats=stats,
        cross_split_excluded=cross_split_excluded,
        unparseable_files=pools["unparseable_files"],
        initial_seed=args.seed,
        final_seed=final_seed,
        attempts=attempts,
        min_val_test_videos=args.min_val_test_videos,
        is_met=is_met,
        ratios=ratios_dict,
        config=config_dict
    )

    # Print formatted console summary table
    print_summary_table(
        stats=stats,
        cross_split_excluded=cross_split_excluded,
        unparseable_files=pools["unparseable_files"],
        initial_seed=args.seed,
        final_seed=final_seed,
        attempts=attempts,
        min_val_test_videos=args.min_val_test_videos,
        is_met=is_met
    )


if __name__ == "__main__":
    main()
