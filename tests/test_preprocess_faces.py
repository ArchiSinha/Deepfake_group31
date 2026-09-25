"""
Unit tests for src/preprocess_faces.py

Tests pure utility functions (box expansion, video ID extraction, atomic save)
and mocked detection pipeline (no GPU required).

Run with: pytest tests/test_preprocess_faces.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

# Import functions under test
from src.preprocess_faces import (
    atomic_save,
    clamp_and_pad,
    compute_video_stats,
    expand_box,
    process_frame,
    video_id_from_name,
)


# ============================================================================
# Test video_id_from_name
# ============================================================================


def test_video_id_from_name_celeb_synthesis():
    """Test video ID extraction from Celeb-synthesis frame."""
    assert video_id_from_name("id5_id1_0003_f2.jpg") == "id5_id1_0003"


def test_video_id_from_name_celeb_real():
    """Test video ID extraction from Celeb-real frame."""
    assert video_id_from_name("id42_0005_f7.jpg") == "id42_0005"


def test_video_id_from_name_youtube():
    """Test video ID extraction from YouTube-real frame."""
    assert video_id_from_name("123_f0.jpg") == "123"


def test_video_id_from_name_no_frame_suffix():
    """Test video ID when no frame suffix present (edge case)."""
    assert video_id_from_name("video_without_suffix.jpg") == "video_without_suffix"


def test_video_id_from_name_multiple_underscores():
    """Test video ID with multiple underscores before frame index."""
    assert video_id_from_name("long_video_name_here_f15.jpg") == "long_video_name_here"


# ============================================================================
# Test expand_box
# ============================================================================


def test_expand_box_centered():
    """Test box expansion centered in frame."""
    box = (100, 100, 200, 200)  # 100x100 box
    expanded = expand_box(box, margin=1.3, frame_shape=(480, 640))
    # Center: (150, 150), side: max(130, 130) = 130
    # New box: (150-65, 150-65, 150+65, 150+65) = (85, 85, 215, 215)
    # No clamping, so result is exact
    assert expanded == (85, 85, 215, 215)


def test_expand_box_rectangular():
    """Test box expansion with non-square box (should become square)."""
    box = (100, 100, 200, 150)  # 100x50 box
    expanded = expand_box(box, margin=1.3, frame_shape=(480, 640))
    # Center: (150, 125), sizes: (130, 65), side: 130
    # New box: (150-65, 125-65, 150+65, 125+65) = (85, 60, 215, 190)
    assert expanded == (85, 60, 215, 190)


def test_expand_box_edge_no_clamp():
    """Test box expansion near edge returns negative coords (no clamping)."""
    box = (10, 10, 60, 60)  # 50x50 box near edge
    expanded = expand_box(box, margin=2.0, frame_shape=(100, 100))
    # Center: (35, 35), side: 100
    # New box: (35-50, 35-50, 35+50, 35+50) = (-15, -15, 85, 85)
    # NOT clamped anymore — returns negative values
    assert expanded == (-15, -15, 85, 85)


def test_expand_box_no_expansion():
    """Test box with margin=1.0 (no expansion, just squaring)."""
    box = (100, 100, 200, 180)  # 100x80 box
    expanded = expand_box(box, margin=1.0, frame_shape=(480, 640))
    # Center: (150, 140), sizes: (100, 80), side: 100
    # New box: (150-50, 140-50, 150+50, 140+50) = (100, 90, 200, 190)
    assert expanded == (100, 90, 200, 190)


# ============================================================================
# Test clamp_and_pad
# ============================================================================


def test_clamp_and_pad_no_padding():
    """Test crop with box entirely within frame."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    box = (100, 100, 300, 300)  # 200x200 box
    crop = clamp_and_pad(frame, box, target_size=224)

    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8


def test_clamp_and_pad_with_edge():
    """Test crop with box extending beyond frame (needs padding)."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    box = (-10, -10, 90, 90)  # Extends 10px beyond left/top edges, 100x100 box
    crop = clamp_and_pad(frame, box, target_size=100)

    assert crop.shape == (100, 100, 3)
    # Padding should replicate edge pixels (all should be 128)
    assert np.all(crop == 128)


def test_clamp_and_pad_upsample():
    """Test upscaling uses INTER_CUBIC."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 64
    box = (10, 10, 30, 30)  # 20x20 box, will upsample to 224
    crop = clamp_and_pad(frame, box, target_size=224)

    assert crop.shape == (224, 224, 3)
    # Values should be close to 64 (cubic interpolation may introduce slight variation)
    assert 60 <= np.mean(crop) <= 68


def test_clamp_and_pad_downsample():
    """Test downscaling uses INTER_AREA."""
    frame = np.ones((500, 500, 3), dtype=np.uint8) * 200
    box = (0, 0, 400, 400)  # 400x400 box, will downsample to 224
    crop = clamp_and_pad(frame, box, target_size=224)

    assert crop.shape == (224, 224, 3)
    # Values should be close to 200
    assert 195 <= np.mean(crop) <= 205


# ============================================================================
# Test atomic_save
# ============================================================================


def test_atomic_save_success():
    """Test atomic save creates output file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dest = tmpdir / "test.jpg"

        atomic_save(image, dest, quality=95)

        assert dest.exists()
        assert dest.stat().st_size > 0

        # Verify it can be read back
        loaded = cv2.imread(str(dest))
        assert loaded is not None
        assert loaded.shape == (224, 224, 3)


def test_atomic_save_creates_parent():
    """Test atomic save creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dest = tmpdir / "subdir" / "nested" / "test.jpg"

        atomic_save(image, dest, quality=95)

        assert dest.exists()
        assert dest.parent.exists()


def test_atomic_save_overwrite():
    """Test atomic save overwrites existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dest = tmpdir / "test.jpg"

        # Write initial file (solid color, highly compressible)
        image1 = np.zeros((224, 224, 3), dtype=np.uint8)
        atomic_save(image1, dest)
        mtime1 = dest.stat().st_mtime

        # Small delay to ensure different mtime
        import time
        time.sleep(0.01)

        # Overwrite with different image (random, less compressible)
        image2 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        atomic_save(image2, dest)
        mtime2 = dest.stat().st_mtime

        assert dest.exists()
        # Verify file was actually rewritten (mtime changed)
        assert mtime2 > mtime1


# ============================================================================
# Test compute_video_stats
# ============================================================================


def test_compute_video_stats_empty():
    """Test video stats with no records."""
    stats = compute_video_stats([])
    assert stats == {}


def test_compute_video_stats_single_video():
    """Test video stats with one video."""
    records = [
        {"video_id": "vid1", "status": "success"},
        {"video_id": "vid1", "status": "success"},
        {"video_id": "vid1", "status": "no_face"},
    ]
    stats = compute_video_stats(records)

    assert "vid1" in stats
    assert stats["vid1"]["total"] == 3
    assert stats["vid1"]["success"] == 2
    assert abs(stats["vid1"]["rate"] - 2/3) < 0.001


def test_compute_video_stats_multiple_videos():
    """Test video stats with multiple videos."""
    records = [
        {"video_id": "vid1", "status": "success"},
        {"video_id": "vid1", "status": "success"},
        {"video_id": "vid2", "status": "success"},
        {"video_id": "vid2", "status": "no_face"},
        {"video_id": "vid2", "status": "no_face"},
        {"video_id": "vid3", "status": "no_face"},
    ]
    stats = compute_video_stats(records)

    assert stats["vid1"]["rate"] == 1.0  # 2/2
    assert abs(stats["vid2"]["rate"] - 1/3) < 0.001  # 1/3
    assert stats["vid3"]["rate"] == 0.0  # 0/1


# ============================================================================
# Test process_frame with mocked detector
# ============================================================================


def test_process_frame_success():
    """Test successful face detection and processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test frame
        frame_path = tmpdir / "test_f0.jpg"
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), frame)

        out_path = tmpdir / "out_f0.jpg"

        # Mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = (
            np.array([[100, 100, 300, 300]]),  # boxes
            np.array([0.95])  # probs
        )

        status, metadata = process_frame(
            frame_path=frame_path,
            out_path=out_path,
            detector=mock_detector,
            min_prob=0.90,
            margin=1.3,
            target_size=224,
            resume=False,
            dry_run=False
        )

        assert status == "success"
        assert metadata["det_prob"] == 0.95
        assert metadata["n_faces"] == 1
        assert metadata["box"] is not None
        assert out_path.exists()


def test_process_frame_no_face():
    """Test frame with no face detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        frame_path = tmpdir / "test_f0.jpg"
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), frame)

        out_path = tmpdir / "out_f0.jpg"

        # Mock detector returning no faces
        mock_detector = Mock()
        mock_detector.detect.return_value = (None, None)

        status, metadata = process_frame(
            frame_path=frame_path,
            out_path=out_path,
            detector=mock_detector,
            min_prob=0.90,
            margin=1.3,
            target_size=224,
            resume=False,
            dry_run=False
        )

        assert status == "no_face"
        assert metadata["det_prob"] is None
        assert metadata["n_faces"] == 0
        assert not out_path.exists()


def test_process_frame_low_confidence():
    """Test frame with detection below confidence threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        frame_path = tmpdir / "test_f0.jpg"
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), frame)

        out_path = tmpdir / "out_f0.jpg"

        # Mock detector with low confidence
        mock_detector = Mock()
        mock_detector.detect.return_value = (
            np.array([[100, 100, 300, 300]]),
            np.array([0.75])  # Below 0.90 threshold
        )

        status, metadata = process_frame(
            frame_path=frame_path,
            out_path=out_path,
            detector=mock_detector,
            min_prob=0.90,
            margin=1.3,
            target_size=224,
            resume=False,
            dry_run=False
        )

        assert status == "low_confidence"
        assert metadata["det_prob"] == 0.75
        assert not out_path.exists()


def test_process_frame_multiple_faces():
    """Test frame with multiple faces (should select largest)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        frame_path = tmpdir / "test_f0.jpg"
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), frame)

        out_path = tmpdir / "out_f0.jpg"

        # Mock detector with two faces (second is larger)
        mock_detector = Mock()
        mock_detector.detect.return_value = (
            np.array([
                [50, 50, 150, 150],    # 100x100 = 10,000
                [200, 200, 450, 450]   # 250x250 = 62,500 (larger)
            ]),
            np.array([0.92, 0.95])
        )

        status, metadata = process_frame(
            frame_path=frame_path,
            out_path=out_path,
            detector=mock_detector,
            min_prob=0.90,
            margin=1.3,
            target_size=224,
            resume=False,
            dry_run=False
        )

        assert status == "success"
        assert metadata["n_faces"] == 2
        assert metadata["det_prob"] == 0.95  # Selected larger face
        assert out_path.exists()


def test_process_frame_resume_skip():
    """Test resume mode skips existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        frame_path = tmpdir / "test_f0.jpg"
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), frame)

        out_path = tmpdir / "out_f0.jpg"

        # Create existing output
        existing = np.ones((224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), existing)

        mock_detector = Mock()

        status, metadata = process_frame(
            frame_path=frame_path,
            out_path=out_path,
            detector=mock_detector,
            min_prob=0.90,
            margin=1.3,
            target_size=224,
            resume=True,  # Resume mode
            dry_run=False
        )

        assert status == "skipped"
        mock_detector.detect.assert_not_called()  # Should not run detector


def test_process_frame_dry_run():
    """Test dry run mode doesn't write files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        frame_path = tmpdir / "test_f0.jpg"
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), frame)

        out_path = tmpdir / "out_f0.jpg"

        mock_detector = Mock()
        mock_detector.detect.return_value = (
            np.array([[100, 100, 300, 300]]),
            np.array([0.95])
        )

        status, metadata = process_frame(
            frame_path=frame_path,
            out_path=out_path,
            detector=mock_detector,
            min_prob=0.90,
            margin=1.3,
            target_size=224,
            resume=False,
            dry_run=True  # Dry run
        )

        assert status == "success"
        assert not out_path.exists()  # Should not write


def test_process_frame_read_error():
    """Test handling of corrupted/unreadable frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create empty/corrupted file
        frame_path = tmpdir / "corrupt.jpg"
        frame_path.write_text("not an image")

        out_path = tmpdir / "out.jpg"
        mock_detector = Mock()

        status, metadata = process_frame(
            frame_path=frame_path,
            out_path=out_path,
            detector=mock_detector,
            min_prob=0.90,
            margin=1.3,
            target_size=224,
            resume=False,
            dry_run=False
        )

        assert status == "read_error"
        mock_detector.detect.assert_not_called()


def test_process_frame_detector_error():
    """Test handling of detector exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        frame_path = tmpdir / "test_f0.jpg"
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), frame)

        out_path = tmpdir / "out_f0.jpg"

        # Mock detector that raises exception
        mock_detector = Mock()
        mock_detector.detect.side_effect = RuntimeError("CUDA OOM")

        status, metadata = process_frame(
            frame_path=frame_path,
            out_path=out_path,
            detector=mock_detector,
            min_prob=0.90,
            margin=1.3,
            target_size=224,
            resume=False,
            dry_run=False
        )

        assert status == "detector_error"
        assert not out_path.exists()


# ============================================================================
# Additional tests for edge padding behavior
# ============================================================================


def test_expand_box_near_origin_negative_coords():
    """Test expand_box on face very close to origin returns negative coords."""
    box = (5, 5, 30, 30)  # 25x25 box near (0, 0)
    expanded = expand_box(box, margin=2.0, frame_shape=(200, 200))
    # Center: (17.5, 17.5), side: 50
    # New box: (17.5-25, 17.5-25, 17.5+25, 17.5+25) = (-7.5, -7.5, 42.5, 42.5)
    # Rounded: (-8, -8, 42, 42) or (-7, -7, 43, 43) depending on rounding
    # Check that x1 and y1 are negative
    assert expanded[0] < 0
    assert expanded[1] < 0
    # Check the box is square
    assert (expanded[2] - expanded[0]) == (expanded[3] - expanded[1])


def test_clamp_and_pad_produces_square_before_resize():
    """Test clamp_and_pad produces square crop via padding, not stretching."""
    # Create a 200x200 frame with a distinct pattern
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:150, 50:150] = 255  # White square in center

    # Box that extends beyond frame bounds: should be 100x100 but goes negative
    box = (-20, -20, 80, 80)  # 100x100 box extending beyond top-left

    # Before clamp_and_pad, the box is 100x100 (square)
    # After padding, it should still be 100x100 (square) before final resize
    crop = clamp_and_pad(frame, box, target_size=224)

    # Final output should be 224x224
    assert crop.shape == (224, 224, 3)

    # To verify it was padded (not stretched), we check that the intermediate
    # crop after padding was square. We can't directly access that, but we can
    # verify the logic by checking a simpler case where target_size = box size
    crop_no_resize = clamp_and_pad(frame, box, target_size=100)
    assert crop_no_resize.shape == (100, 100, 3)  # Should be exactly square
